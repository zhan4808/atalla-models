"""AtallaC kernel generators.

Each module exposes a function that returns parameterised AtallaC source code
for a single kernel type. The generated C is compiled by ppci into Atalla
assembly and then encoded into .in files for the functional simulator.
"""

from kernels.common import ADDR_TABLE, TILE, sdma_ctl_val, sdma_ctl_expr
from kernels.gemm import gemm_c
from kernels.relu import relu_c
from kernels.softmax import softmax_c
from kernels.maxpool import maxpool_c
from kernels.add import add_c
