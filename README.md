# Meent
Meent is a RCWA solver and its applications on optimization problem. We are expecting that this tool can accelerate ML research in photonics.

How to install
---
You can install from PyPI
```shell
pip install meent
```
or download this repo and run
```shell
pip install .
```

## How to use
Meent provides Numpy, JAX and PyTorch as a backend.

```python
import meent
from meent.rcwa import call_solver

# mode 0 = Numpy
# mode 1 = JAX
# mode 2 = PyTorch

mode_key = 1
solver = meent.rcwa.call_solver(mode=mode_key, ...)
```

## When to use


|                 | Numpy | JAX  | PyTorch |
|-----------------|:-----:|:----:|:-------:|
| 64bit support   |   O   |  O   |    O    |
| 32bit support   |   O   |  O   |    O    |
| GPU support     |   X   |  O   |    O    |
| TPU support     |   X   |  X   |    X    |
| AD support      |   X   |  O   |    O    |
| Parallelization |   X   |  O   |    X    |



Numpy is simple and light to use. Suggested as a baseline with small ~ medium scale optics problem.  
JAX and PyTorch is recommended for cases having large scale or optimization part.  
If you want parallelized computing with multiple devices(e.g., GPUs), JAX is ready for that.  
But since JAX does jit compilation, it takes much time at the first run. 


# Reference
Many literatures and codes are referred for development and quality assurance

[1] https://opg.optica.org/josa/abstract.cfm?uri=josa-71-7-811, Rigorous coupled-wave analysis of planar-grating diffraction \
[2] https://opg.optica.org/josaa/abstract.cfm?uri=josaa-12-5-1068 \
[3] https://opg.optica.org/josaa/abstract.cfm?uri=josaa-12-5-1077 \
[4] https://opg.optica.org/josaa/abstract.cfm?uri=josaa-13-5-1019 \
[5] https://opg.optica.org/josaa/abstract.cfm?uri=josaa-13-4-779 \
[6] https://opg.optica.org/josaa/abstract.cfm?uri=josaa-13-9-1870 \
[7] https://empossible.net/emp5337/ \
[8] https://github.com/zhaonat/Rigorous-Coupled-Wave-Analysis (see also our fork: https://github.com/yonghakim/zhaonat-rcwa) \
[9] https://arxiv.org/abs/2101.00901