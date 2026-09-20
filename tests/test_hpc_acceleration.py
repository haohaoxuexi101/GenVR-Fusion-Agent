import numpy as np
import torch

from gmc.mc_cell_gpu import generate_boundary_dataset_torch, generate_internal_dataset_torch
from gmc.response_operator_ultra import UltraPhaseSpace, _project_one_output, _project_outputs_vectorized, _input_condition
from gmc.transport2d import BoundaryGMCSampler
from gmc.model import integrate_rk4_from_latent_cached, integrate_rk4_from_latent_prepared, prepare_condition_cache, gather_condition_cache


def test_vectorized_random_flight_physical_constraints():
    b=generate_boundary_dataset_torch(2048,seed=91,backend='cpu',chunk_size=2048,dtype='float64',progress=False)
    assert b.conditions.shape==(2048,5); assert b.targets.shape==(2048,4)
    assert np.all(np.isfinite(b.targets)); assert np.all((b.targets[:,0]>=0)&(b.targets[:,0]<1))
    assert np.all(b.targets[:,3]>0); assert np.all(b.targets[:,1]**2+b.targets[:,2]**2<=1.00001)
    i=generate_internal_dataset_torch(2048,seed=92,backend='cpu',chunk_size=2048,dtype='float64',progress=False)
    assert np.all(np.isfinite(i.targets)); assert np.all(i.targets[:,3]>0)


def test_square_canonical_condition_dedup_is_exact_fourfold():
    d=UltraPhaseSpace(4,4,16); raw=[]
    for i in range(d.n_state): raw.append(_input_condition(d,i,1.0,0.25,0.25)[0][0])
    assert d.n_state==256
    assert len(np.unique(np.asarray(raw),axis=0))==64


def test_vectorized_projection_matches_scalar():
    rng=np.random.default_rng(7); d=UltraPhaseSpace(4,4,16); dx,dy=1.3,.7
    for face in ('left','right','bottom','top'):
        dxc,dyc=(dx,dy) if face in ('left','right') else (dy,dx)
        n=64; p=rng.random(n); u=rng.uniform(-.7,.7,n); v=rng.uniform(-.7,.7,n)
        r2=u*u+v*v; bad=r2>.9
        if np.any(bad):
            sc=np.sqrt(.9/r2[bad]); u[bad]*=sc; v[bad]*=sc
        idx,w=_project_outputs_vectorized(d,face,p,u,v,dxc,dyc,dx,dy)
        assert np.max(np.abs(w.sum(axis=1)-1.0))<1e-12
        for m in range(n):
            a={j:ww for j,ww in _project_one_output(d,face,p[m],u[m],v[m],dxc,dyc,dx,dy)}
            b={}
            for j,ww in zip(idx[m],w[m]):
                if ww: b[int(j)]=b.get(int(j),0.0)+float(ww)
            assert max(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))<2e-12


def test_prepared_condition_cache_is_same_as_repeated_cache():
    # Random initialized network has zero output layer; perturb it so this is a real algebra test.
    from gmc.model import BoundaryVelocityNet,ModelConfig
    torch.manual_seed(3); m=BoundaryVelocityNet(ModelConfig(c_dim=5,y_dim=4,hidden_dim=32,depth=2)).eval()
    with torch.no_grad(): m.out.weight.normal_(0,.05); m.out.bias.normal_(0,.01)
    cu=torch.randn(5,5); ids=torch.randint(0,5,(96,)); c=cu[ids]; z=torch.randn(96,4)
    a=integrate_rk4_from_latent_cached(m,c,z,4)
    cache=gather_condition_cache(prepare_condition_cache(m,cu),ids)
    b=integrate_rk4_from_latent_prepared(m,cache,z,4)
    assert torch.max(torch.abs(a-b)).item()<2e-6
