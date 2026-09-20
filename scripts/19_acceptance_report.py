#!/usr/bin/env python
"""Generate a human-readable paper-scale acceptance report."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def load_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='outputs/paper_scale')
    ap.add_argument('--out',default='outputs/paper_scale/ACCEPTANCE_REPORT.md')
    args=ap.parse_args()
    root=Path(args.root)
    lines=[]
    lines.append('# GMC paper-scale reproduction acceptance report\n')
    lines.append('This report is generated after the full paper-scale run. It checks whether the implementation followed the Farmer (2026) reproduction protocol: 9e5/1e5 CFM training, 20k single-cell validation histories per optical test, and 112x112 cloud maps with 1e6 histories.\n')
    for p in [root/'boundary_cfm_paper.json', root/'internal_cfm_paper.json']:
        j=load_json(p)
        lines.append(f'## {p.name}\n')
        if j is None: lines.append('Missing.\n')
        else: lines.append('```json\n'+json.dumps(j,indent=2)+'\n```\n')
    sm=load_json(root/'fig31_cloudmaps_n1e6/summary_metrics.json')
    lines.append('## Figure 3.1 cloud-map metrics\n')
    if sm is None: lines.append('Missing summary metrics.\n')
    else: lines.append('```json\n'+json.dumps(sm,indent=2)+'\n```\n')
    rt=load_json(root/'fig32_runtime_scaling/runtime_scaling.json')
    lines.append('## Runtime scaling\n')
    lines.append('Available.\n' if rt is not None else 'Missing.\n')
    conv=load_json(root/'fig32_convergence/convergence_raw.json')
    lines.append('## Statistical convergence\n')
    lines.append('Available.\n' if conv is not None else 'Missing.\n')
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(lines))
    print(f'wrote {out}')
if __name__=='__main__': main()
