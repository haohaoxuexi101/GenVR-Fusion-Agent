#!/usr/bin/env python3
import sys, platform
import numpy as np
import scipy
import matplotlib
import numba
import torch
print('python:', sys.version.replace('\n',' '))
print('platform:', platform.platform())
print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('matplotlib:', matplotlib.__version__)
print('numba:', numba.__version__)
print('torch:', torch.__version__)
print('torch cuda available:', torch.cuda.is_available())
print('torch cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    p=torch.cuda.get_device_properties(0)
    print('gpu memory GiB:', round(p.total_memory/1024**3, 2))
