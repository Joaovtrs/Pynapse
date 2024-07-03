import numpy as np
from brain import Camada, RedeNeural
from brain.funcs_ativacao import FuncReLU, FuncSoftmax
from brain.funcs_custo import FuncNegLogLikelihood

nn = RedeNeural(
    [
        Camada(3, 3, FuncReLU()),
        Camada(3, 3, FuncReLU()),
        Camada(3, 3, FuncSoftmax()),
    ]
)

print(nn)
nn.mutacao(0.5)
