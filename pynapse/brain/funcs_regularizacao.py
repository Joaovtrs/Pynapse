import numpy as np


class FuncRegularizacaoGenerica:
    def __init__(self, nome):
        self.nome = nome

    def __str__(self):
        return self.nome

    def __repr__(self):
        return self.__str__()


class FuncL1(FuncRegularizacaoGenerica):
    def __init__(self, gama):
        super().__init__('Função de Regularização L1')
        self.gama = gama

    def __call__(self, pesos, n_amostras):
        return (
            self.gama
            / n_amostras
            * np.sum([np.sum(np.abs(peso)) for peso in pesos])
        )

    def derivada(self, pesos, n_amostras):
        pesos = [np.where(peso < 0, -1, peso) for peso in pesos]

        return (
            self.gama
            / n_amostras
            * np.array([np.where(peso > 0, 1, peso) for peso in pesos])
        )


class FuncL2(FuncRegularizacaoGenerica):
    def __init__(self, gama):
        super().__init__('Função de Regularização L2')
        self.gama = gama

    def __call__(self, pesos, n_amostras):
        return self.gama / n_amostras / 2 * np.sum(pesos**2)

    def derivada(self, pesos, n_amostras):
        return self.gama / n_amostras * pesos
