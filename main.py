import numpy as np

from brain import Camada, RedeNeural, CamadaGenerica
from brain.funcs_ativacao import FuncReLU, FuncSigmoid, FuncSoftmax
from brain.funcs_custo import FuncNegLogLikelihood

x = np.array([[0.1, 0.2, 0.7]])
y = np.array([[1, 0, 0]])

nn = RedeNeural(
    [
        CamadaGenerica(3),
        Camada(3, 3, FuncReLU()),
        Camada(3, 3, FuncSigmoid()),
        Camada(3, 3, FuncSoftmax()),
    ]
)

nn.camadas[1].pesos = np.array(
    [[0.1, 0.2, 0.3], [0.3, 0.2, 0.7], [0.4, 0.3, 0.9]]
    )
nn.camadas[1].bias = np.array([[1, 1, 1]])
nn.camadas[2].pesos = np.array(
    [[0.2, 0.3, 0.5], [0.3, 0.5, 0.7], [0.6, 0.4, 0.8]]
    )
nn.camadas[2].bias = np.array([[1, 1, 1]])
nn.camadas[3].pesos = np.array(
    [[0.1, 0.4, 0.8], [0.3, 0.7, 0.2], [0.5, 0.2, 0.9]]
    )
nn.camadas[3].bias = np.array([[1, 1, 1]])

print(nn)

for i in range(301):
    custo = nn.backprop(x, y, FuncNegLogLikelihood(), 0.01)

    if i % 30 == 0:
        print(custo)

print(nn)
