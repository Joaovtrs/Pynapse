import numpy as np


class FuncCustoGenerica:
    def __init__(self, nome):
        self.nome = nome

    def __str__(self):
        return self.nome

    def __repr__(self):
        return self.__str__()


class FuncMAE(FuncCustoGenerica):
    def __init__(self):
        super().__init__('Função Média dos Erros Absolutos')

    def __call__(self, y, y_pred):
        return np.mean(np.abs(y - y_pred))

    @staticmethod
    def derivada(y, y_pred):
        return np.sum(np.where(y >= y_pred, 1, -1)) / y.shape[0]


class FuncMSE(FuncCustoGenerica):
    def __init__(self):
        super().__init__('Função Média dos Erros Quadrados')

    def __call__(self, y, y_pred):
        return np.mean((y - y_pred) ** 2) / 2

    @staticmethod
    def derivada(y, y_pred):
        return -(y - y_pred) / y.shape[0]


class FuncSAE(FuncCustoGenerica):
    def __init__(self):
        super().__init__('Função Soma dos Erros Absolutos')

    def __call__(self, y, y_pred):
        return np.sum(np.abs(y - y_pred))

    @staticmethod
    def derivada(y, y_pred):
        return np.sum(np.where(y >= y_pred, 1, -1))


class FuncSSE(FuncCustoGenerica):
    def __init__(self):
        super().__init__('Função Soma dos Erros Quadrados')

    def __call__(self, y, y_pred):
        return np.sum((y - y_pred) ** 2) / 2

    @staticmethod
    def derivada(y, y_pred):
        return -(y - y_pred)


class FuncEntropiaCruzada(FuncCustoGenerica):
    def __init__(self):
        super().__init__('Função Entropia Cruzada')

    def __call__(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    @staticmethod
    def derivada(y, y_pred):
        return -(y - y_pred) / (y_pred * (1 - y_pred) * y.shape[0])


class FuncNegLogLikelihood(FuncCustoGenerica):
    def __init__(self):
        super().__init__('Função Neg. Log-Likelihood')

    def __call__(self, y, y_pred):
        k = np.nonzero(y)
        return np.mean(-np.log(y_pred[k]))

    @staticmethod
    def derivada(y, y_pred):
        return -(y - y_pred) / y.shape[0]
