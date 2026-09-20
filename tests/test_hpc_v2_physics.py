import numpy as np
from gmc.transforms import BoundaryTransform, InternalTransform


def _boundary_data(n=2000, seed=1):
    rng=np.random.default_rng(seed)
    W=rng.uniform(.02,5,n); H=rng.uniform(.02,5,n); yin=rng.uniform(0,H)
    u=rng.uniform(.1,.95,n); v=rng.uniform(-.6,.6,n)
    # choose right-edge exits for a simple guaranteed chord, then encode perimeter p
    x=np.full(n,0.0); y=yin
    xo=W; yo=np.clip(yin+rng.normal(0,.15,n),0,H)
    # perimeter coordinate: bottom W, right follows; s=W+yo
    p=(W+yo)/(2*(W+H))
    chord=np.hypot(xo,yo-yin); L=chord+rng.exponential(.7,n)
    c=np.column_stack([W,H,yin,u,v]).astype(np.float32)
    t=np.column_stack([p,np.full(n,.5),np.zeros(n),L]).astype(np.float32)
    return c,t,chord


def test_boundary_excess_chord_roundtrip_and_constraint():
    c,t,_=_boundary_data()
    tr=BoundaryTransform.fit(c,t,path_mode='excess_chord')
    enc=tr.encode_targets(t,c); dec=tr.decode_targets_np(enc,c)
    # Transform is near-exact away from robust clipping.
    assert np.max(np.abs(dec[:,0]-t[:,0])) < 2e-5
    # Most importantly: decoded total path can never be shorter than its decoded chord.
    chord=tr._path_baseline_np(c,dec[:,0])
    assert np.all(dec[:,3] >= chord)


def test_internal_excess_chord_constraint():
    rng=np.random.default_rng(2); n=1500
    W=rng.uniform(.05,3,n); H=rng.uniform(.05,3,n); x=rng.uniform(0,W); y=rng.uniform(0,H)
    u=rng.uniform(-.8,.8,n); v=rng.uniform(-.8,.8,n)
    # top edge exit
    xo=np.clip(x+rng.normal(0,.1,n),0,W); yo=H
    p=(2*W+H+(W-xo))/(2*(W+H))
    chord=np.hypot(xo-x,yo-y); L=chord+rng.exponential(.5,n)
    c=np.column_stack([W,H,x,y,u,v]).astype(np.float32)
    t=np.column_stack([p,np.zeros(n),np.full(n,.5),L]).astype(np.float32)
    tr=InternalTransform.fit(c,t,path_mode='excess_chord')
    dec=tr.decode_targets_np(tr.encode_targets(t,c),c)
    assert np.all(dec[:,3] >= tr._path_baseline_np(c,dec[:,0]))


def test_legacy_transform_backward_compatible():
    c,t,_=_boundary_data(100)
    tr=BoundaryTransform.fit(c,t,path_mode='absolute')
    d=tr.to_dict(); d.pop('path_mode')
    old=BoundaryTransform.from_dict(d)
    assert old.path_mode=='absolute'
    enc=old.encode_targets(t)
    dec=old.decode_targets_np(enc)
    assert np.all(np.isfinite(dec))
