import numpy as np

from .funcs_ativacao import FuncLinear


class Camada:
    def __init__(self, n_entrada, n_neuronios, func_ativacao=FuncLinear()):
        self.input = None
        self.pesos = np.random.random((n_neuronios, n_entrada))
        self.bias = np.random.random(n_neuronios)
        self.func_ativacao = func_ativacao

    def __str__(self):
        return (f'{self.pesos.shape[1]} entradas e {self.pesos.shape[0]} '
                f'neurônios\nPesos:\n{self.pesos}\nBias: {self.bias}\nFunção '
                f'de Ativação: {self.func_ativacao}')

    def __repr__(self):
        return self.__str__()

    def __call__(self, x):
        return self.func_ativacao(np.dot(x, self.pesos.T) + self.bias)

    def backprop(self, x, dout, lr):
        inp = np.dot(x, self.pesos.T) + self.bias
        dinp = self.func_ativacao.derivada(inp) * dout

        dpesos = np.dot(dinp.T, x)
        dbias = dinp.sum(axis=0)
        dx = np.dot(dinp, self.pesos)

        self.pesos = self.pesos - lr * dpesos
        self.bias = self.bias - lr * dbias

        return dx
