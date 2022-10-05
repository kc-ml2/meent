# meent
Meent is a RCWA solver and its applications on optimization problem. We are expecting that this tool can accelerate ML research in photonics.

run examples/ex1_get_spectrum.py

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

How to use
------
Now meent provides 2 modes for RCWA solver - light mode and optimization mode. 
Light mode uses pure numpy while opt mode uses JAX.
Didn't perform well-structured test but light mode seems way faster than opt mode so choose mode according to your goal.


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