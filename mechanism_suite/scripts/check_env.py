#!/usr/bin/env python3
import sys, platform
print('python', sys.version.replace('\n',' '))
print('platform', platform.platform())
try:
    import torch, torchvision
    print('torch', torch.__version__)
    print('torchvision', torchvision.__version__)
    print('cuda available', torch.cuda.is_available())
    print('cuda runtime', torch.version.cuda)
    if torch.cuda.is_available():
        print('gpu count', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            print(f'gpu[{i}] {p.name} memory={p.total_memory/2**30:.1f} GiB')
except Exception as e:
    print('TORCH CHECK FAILED:', repr(e)); raise
for pkg in ['numpy','scipy','sklearn','matplotlib','datasets','PIL','transformers']:
    try:
        m=__import__(pkg); print(pkg, getattr(m,'__version__','ok'))
    except Exception as e:
        print(pkg,'MISSING',repr(e)); raise
