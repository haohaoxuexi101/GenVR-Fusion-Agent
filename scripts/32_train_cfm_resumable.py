#!/usr/bin/env python
"""Resumable Farmer-style CFM trainer for boundary/internal single-cell samplers.

The important feature is that OneCycleLR is parameterized by the *planned total*
number of epochs.  Short invocations may resume optimizer/scheduler state without
silently restarting the learning-rate schedule.  This makes long paper-scale runs
robust to batch-system wall times and is also convenient for CPU verification.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if not torch.cuda.is_available():
    torch.set_num_threads(1)

from gmc.mc_cell import load_npz, load_internal_npz
from gmc.model import BoundaryVelocityNet, ModelConfig, cfm_loss, cfm_loss_fixed
from gmc.transforms import BoundaryTransform, InternalTransform


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['boundary','internal'], required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--n-train', type=int, required=True)
    ap.add_argument('--n-val', type=int, required=True)
    ap.add_argument('--total-epochs', type=int, default=100)
    ap.add_argument('--chunk-epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--hidden-dim', type=int, default=160)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--transform-qlo', type=float, default=1e-5)
    ap.add_argument('--transform-qhi', type=float, default=1.0-1e-5)
    ap.add_argument('--path-mode', choices=['absolute','excess-chord'], default='excess-chord',
                    help='recommended: learn non-negative path excess over entry-exit chord; old checkpoints use absolute')
    ap.add_argument('--onecycle-pct-start', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--amp', choices=['off','bf16','fp16'], default='off')
    ap.add_argument('--compile', action='store_true', help='torch.compile the velocity network')
    ap.add_argument('--matmul-precision', choices=['highest','high','medium'], default='highest')
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.matmul_precision != 'highest'
        torch.backends.cudnn.allow_tf32 = args.matmul_precision != 'highest'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    last_path = run/'last.pt'; best_path = run/'best.pt'; hist_path=run/'history.json'

    batch = load_npz(args.data) if args.kind=='boundary' else load_internal_npz(args.data)
    n_total=args.n_train+args.n_val
    if len(batch.conditions)<n_total: raise ValueError(f'data has {len(batch.conditions)}, need {n_total}')
    C=batch.conditions[:n_total]; Y=batch.targets[:n_total]
    c_tr_raw,c_va_raw=C[:args.n_train],C[args.n_train:n_total]
    y_tr_raw,y_va_raw=Y[:args.n_train],Y[args.n_train:n_total]

    if args.resume and last_path.exists():
        state=torch.load(last_path,map_location='cpu',weights_only=False)
        if state['kind']!=args.kind: raise ValueError('checkpoint kind mismatch')
        transform=(BoundaryTransform.from_dict(state['transform']) if args.kind=='boundary' else InternalTransform.from_dict(state['transform']))
        requested_pm='excess_chord' if args.path_mode=='excess-chord' else 'absolute'
        if getattr(transform,'path_mode','absolute') != requested_pm:
            raise ValueError(f"resume path-mode mismatch: checkpoint={getattr(transform,'path_mode','absolute')} requested={requested_pm}")
    else:
        pm='excess_chord' if args.path_mode=='excess-chord' else 'absolute'
        transform=(BoundaryTransform.fit(c_tr_raw,y_tr_raw,args.transform_qlo,args.transform_qhi,path_mode=pm) if args.kind=='boundary'
                   else InternalTransform.fit(c_tr_raw,y_tr_raw,args.transform_qlo,args.transform_qhi,path_mode=pm))

    c_tr=transform.encode_conditions(c_tr_raw); y_tr=transform.encode_targets(y_tr_raw,c_tr_raw)
    c_va=transform.encode_conditions(c_va_raw); y_va=transform.encode_targets(y_va_raw,c_va_raw)
    ds=TensorDataset(torch.from_numpy(c_tr),torch.from_numpy(y_tr))
    gen=torch.Generator(); gen.manual_seed(args.seed+177)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,drop_last=True,generator=gen,
        num_workers=max(0,args.num_workers),pin_memory=(device.type=='cuda'),
        persistent_workers=(args.num_workers>0),prefetch_factor=(4 if args.num_workers>0 else None))
    if len(loader)==0: raise ValueError('batch size larger than training set')

    val_c=torch.from_numpy(c_va).to(device); val_y=torch.from_numpy(y_va).to(device)
    # Deterministic validation sample and deterministic validation subset.
    gv=torch.Generator(device=device); gv.manual_seed(args.seed+99173)
    val_eps=torch.randn(val_y.shape,generator=gv,device=device,dtype=val_y.dtype)
    val_t=torch.rand((val_y.shape[0],1),generator=gv,device=device,dtype=val_y.dtype)
    if val_y.shape[0]>50_000:
        gi=torch.Generator(device=device); gi.manual_seed(args.seed+541)
        val_idx=torch.randperm(val_y.shape[0],generator=gi,device=device)[:50_000]
    else:
        val_idx=torch.arange(val_y.shape[0],device=device)

    cfg=ModelConfig(c_dim=5 if args.kind=='boundary' else 6,y_dim=4,hidden_dim=args.hidden_dim,depth=args.depth)
    base_model=BoundaryVelocityNet(cfg).to(device)
    model=torch.compile(base_model,mode='reduce-overhead') if args.compile else base_model
    adamw_kwargs=dict(lr=args.lr,betas=(0.9,0.99),weight_decay=args.weight_decay)
    if device.type=='cuda':
        adamw_kwargs['fused']=True
    try:
        opt=torch.optim.AdamW(base_model.parameters(),**adamw_kwargs)
    except (TypeError,RuntimeError):
        adamw_kwargs.pop('fused',None); opt=torch.optim.AdamW(base_model.parameters(),**adamw_kwargs)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=args.lr,total_steps=args.total_epochs*len(loader),pct_start=args.onecycle_pct_start,anneal_strategy='cos')
    start_epoch=0; best_val=float('inf'); history=[]

    if args.resume and last_path.exists():
        state=torch.load(last_path,map_location=device,weights_only=False)
        # Require schedule-defining quantities to be compatible.
        old=state.get('train_config',{})
        for key,cur in [('total_epochs',args.total_epochs),('steps_per_epoch',len(loader)),('batch_size',args.batch_size)]:
            if key in old and int(old[key])!=int(cur): raise ValueError(f'resume mismatch {key}: {old[key]} vs {cur}')
        base_model.load_state_dict(state['model_state']); opt.load_state_dict(state['optimizer_state']); sched.load_state_dict(state['scheduler_state'])
        start_epoch=int(state['epoch']); best_val=float(state.get('best_val_loss',float('inf'))); history=state.get('history',[])
        print(f'resumed epoch={start_epoch}, best_val={best_val:.6e}, scheduler_step={sched.last_epoch}',flush=True)

    end_epoch=min(args.total_epochs,start_epoch+args.chunk_epochs)
    if start_epoch>=args.total_epochs:
        print('training already complete'); return

    for epoch in range(start_epoch+1,end_epoch+1):
        model.train(); runloss=0.0
        amp_dtype = torch.bfloat16 if args.amp=='bf16' else torch.float16
        use_amp = device.type=='cuda' and args.amp!='off'
        scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and args.amp=='fp16'))
        for cb,yb in loader:
            cb=cb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=amp_dtype,enabled=use_amp):
                loss=cfm_loss(model,cb,yb)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            sched.step(); runloss+=float(loss.detach().cpu())
        train_loss=runloss/len(loader)
        model.eval()
        with torch.no_grad():
            vi=val_idx
            amp_dtype = torch.bfloat16 if args.amp=='bf16' else torch.float16
            with torch.autocast(device_type='cuda',dtype=amp_dtype,enabled=(device.type=='cuda' and args.amp!='off')):
                vl=cfm_loss_fixed(model,val_c[vi],val_y[vi],val_t[vi],val_eps[vi])
            val_loss=float(vl.float().cpu())
        lr=float(sched.get_last_lr()[0])
        row={'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss,'lr':lr}; history.append(row)
        print(f'epoch {epoch:04d}/{args.total_epochs} train={train_loss:.6e} val={val_loss:.6e} lr={lr:.3e}',flush=True)

        train_config={**vars(args),'steps_per_epoch':len(loader)}
        state={
            'kind':args.kind,'epoch':epoch,'best_val_loss':min(best_val,val_loss),'model_state':base_model.state_dict(),
            'model_config':cfg.to_dict(),'transform':transform.to_dict(),'optimizer_state':opt.state_dict(),
            'scheduler_state':sched.state_dict(),'train_config':train_config,'history':history,
            'train_args':train_config,'model_kind':args.kind,
        }
        torch.save(state,last_path)
        if val_loss<best_val:
            best_val=val_loss
            state['best_val_loss']=best_val
            torch.save(state,best_path)
            print(f'saved best -> {best_path}',flush=True)
        hist_path.write_text(json.dumps({'kind':args.kind,'best_val_loss':best_val,'history':history,'train_config':train_config},indent=2),encoding='utf-8')

    print(f'chunk complete: epoch {end_epoch}/{args.total_epochs}; best={best_val:.6e}',flush=True)

if __name__=='__main__': main()
