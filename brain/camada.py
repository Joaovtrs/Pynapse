import numpy as np

from .funcs_ativacao import FuncLinear


class Camada:
    def __init__(self, n_entrada, n_neuronios, func_ativacao=FuncLinear()):
        self.pesos = np.random.random((n_neuronios, n_entrada))
        self.bias = np.random.random(n_neuronios)
        self.func_ativacao = func_ativacao

        self.x = None
        self.input, self.out = None, None

        self.dinput, self.dx = None, None
        self.dpesos, self.dbias = None, None

    def __str__(self):
        return (f'{self.pesos.shape[1]} entradas e {self.pesos.shape[0]} '
                f'neurônios\nPesos:\n{self.pesos}\nBias: {self.bias}\nFunção '
                f'de Ativação: {self.func_ativacao}')

    def __repr__(self):
        return self.__str__()

    def __call__(self, x):
        self.x = x
        self.input = np.dot(x, self.pesos.T) + self.bias
        self.out = self.func_ativacao(self.input)
        return self.out

    def backprop(self, dout):
        self.dinput = self.func_ativacao.derivada(self.input) * dout

        self.dpesos = np.dot(self.dinput.T, self.x)
        self.dbias = self.dinput.sum(axis=0)
        self.dx = np.dot(self.dinput, self.pesos)

        return self.dx

    def atualizar(self, lr):
        self.pesos = self.pesos - lr * self.dpesos
        self.bias = self.bias - lr * self.dbias
