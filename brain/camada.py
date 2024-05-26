import numpy as np

from .funcs_ativacao import FuncLinear
from .funcs_init_pesos import FuncRandomNormal, FuncUns


class CamadaGenerica:
    def __init__(self, n_entrada):
        self.n_entrada = n_entrada

    def __str__(self):
        return 'Camada genérica'

    def __repr__(self):
        return self.__str__()

    def __call__(self, x):
        return x

    def backprop(self, dout):
        return dout

    def atualizar(self, lr):
        pass


class Camada(CamadaGenerica):
    def __init__(
        self,
        n_entrada,
        n_neuronios,
        func_ativacao=FuncLinear(),
        init_pesos=FuncRandomNormal(),
        init_bias=FuncUns()
    ):
        super().__init__(n_entrada)
        self.pesos = init_pesos(n_neuronios, n_entrada)
        self.bias = init_bias(1, n_neuronios)
        self.func_ativacao = func_ativacao

        self.x = None
        self.activ_inp, self.out = None, None

        self.dinp, self.dx = None, None
        self.dpesos, self.dbias = None, None

    def __str__(self):
        return (f'{self.pesos.shape[1]} entradas e {self.pesos.shape[0]} '
                f'neurônios\nPesos:\n{self.pesos}\nBias: {self.bias}\nFunção '
                f'de Ativação: {self.func_ativacao}')

    def __call__(self, x):
        self.x = x
        self.activ_inp = np.dot(x, self.pesos.T) + self.bias
        self.out = self.func_ativacao(self.activ_inp)
        return self.out

    def backprop(self, dout):
        self.dinp = self.func_ativacao.derivada(self.activ_inp) * dout

        self.dpesos = np.dot(self.dinp.T, self.x)
        self.dbias = self.dinp.sum(axis=0, keepdims=True)
        self.dx = np.dot(self.dinp, self.pesos)

        return self.dx

    def atualizar(self, lr):
        self.pesos = self.pesos - lr * self.dpesos
        self.bias = self.bias - lr * self.dbias
