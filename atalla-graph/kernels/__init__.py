"""AtallaC kernel generators.

Each module exposes a function that returns parameterised AtallaC source code
for a single kernel type. The generated C is compiled by ppci into Atalla
assembly and then encoded into .in files for the functional simulator.
"""

from .common import ADDR_TABLE, TILE, sdma_ctl_val, sdma_ctl_expr
from .gemm import gemm_c
from .relu import relu_c
from .softmax import softmax_c, softmax_c_batched
from .maxpool import maxpool_c
from .add import add_c
from .mul import mul_c
from .layernorm import layernorm_c
