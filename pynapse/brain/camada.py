from random import random

import numpy as np

from .funcs_ativacao import FuncLinear
from .funcs_init_pesos import FuncRandomNormal, FuncUns
from .funcs_regularizacao import FuncL2


class Camada:
    def __init__(
        self,
        n_entrada,
        n_neuronios,
        func_ativacao=FuncLinear(),
        func_regularizacao=FuncL2(0),
        init_pesos=FuncRandomNormal(),
        init_bias=FuncUns(),
    ):
        self.pesos = init_pesos(n_neuronios, n_entrada)
        self.bias = init_bias(1, n_neuronios)
        self.func_ativacao = func_ativacao
        self.func_regularizacao = func_regularizacao

        self.x = None
        self.activ_inp, self.out = None, None

        self.dinp, self.dx = None, None
        self.dpesos, self.dbias = None, None
        self.dpesos_anterior = 0.0

        self.dropout_mask = None

    def __str__(self):
        return (
            f'\t\t{self.pesos.shape[1]} entradas e '
            + f'{self.pesos.shape[0]} neurônios'
            + f'\n\t\tFunção de ativação: {self.func_ativacao}'
            + f'\n\t\tNúmero de parâmetros: {self.get_paran_cout()}'
        )

    def __repr__(self):
        return self.__str__()

    def __call__(self, x, dropout_prob=0):
        self.x = x
        self.activ_inp = np.dot(x, self.pesos.T) + self.bias
        self.dropout_mask = np.random.binomial(
            1, 1 - dropout_prob, self.activ_inp.shape
        ) / (1 - dropout_prob)
        self.out = self.func_ativacao(self.activ_inp) * self.dropout_mask

        return self.out

    def backprop(self, dout):
        self.dinp = (
            self.func_ativacao.derivada(self.activ_inp)
            * dout
            * self.dropout_mask
        )

        self.dpesos = np.dot(self.dinp.T, self.x)
        self.dpesos = self.dpesos + self.func_regularizacao.derivada(
            self.pesos, self.x.shape[0]
        )
        self.dbias = self.dinp.sum(axis=0, keepdims=True)
        self.dx = np.dot(self.dinp, self.pesos)

        return self.dx

    def atualizar(self, lr, momentum=0):
        self.dpesos_anterior = (
            momentum * self.dpesos_anterior + lr * self.dpesos
        )
        self.pesos = self.pesos - self.dpesos_anterior
        self.bias = self.bias - lr * self.dbias

    def get_paran_cout(self):
        return self.pesos.shape[1] * self.pesos.shape[0] + self.bias.shape[1]

    def mutacao(self, prob_mutacao=0.01, forca=0.01):
        mutacoes = np.random.binomial(1, prob_mutacao, self.pesos.shape)
        mutacoes = mutacoes * np.random.randn(*self.pesos.shape) * forca
        self.pesos = self.pesos + mutacoes

        mutacoes = np.random.binomial(1, prob_mutacao, self.bias.shape)
        mutacoes = mutacoes * np.random.randn(*self.bias.shape) * forca
        self.bias = self.bias + mutacoes
