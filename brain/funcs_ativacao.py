import numpy as np


class FuncAtivacaoGenerica:
    def __init__(self, nome):
        self.nome = nome

    def __str__(self):
        return self.nome

    def __repr__(self):
        return self.__str__()


class FuncLinear(FuncAtivacaoGenerica):
    def __init__(self):
        super().__init__('Função Linear')

    def __call__(self, x):
        return x

    @staticmethod
    def derivada(x):
        return np.ones_like(x)


class FuncSigmoid(FuncAtivacaoGenerica):
    def __init__(self):
        super().__init__('Função Sigmoid')

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada(self, x):
        y = self.__call__(x)
        return y * (1 - y)


class FuncTanh(FuncAtivacaoGenerica):
    def __init__(self):
        super().__init__('Função Tanh')

    def __call__(self, x):
        expx = np.exp(x)
        expmx = np.exp(-x)
        return (expx - expmx) / (expx + expmx)

    def derivada(self, x):
        y = self.__call__(x)
        return 1 - y ** 2


class FuncReLU(FuncAtivacaoGenerica):
    def __init__(self):
        super().__init__('Função ReLU')

    def __call__(self, x):
        return np.maximum(0, x)

    @staticmethod
    def derivada(x):
        return np.where(x <= 0, 0, 1)


class FuncLeakyReLU(FuncAtivacaoGenerica):
    def __init__(self, alfa):
        super().__init__('Função Leaky ReLU')
        self.alfa = alfa

    def __call__(self, x):
        return np.where(x <= 0, self.alfa * x, x)

    def derivada(self, x):
        return np.where(x <= 0, self.alfa, 1)


class FuncELU(FuncAtivacaoGenerica):
    def __init__(self, alfa):
        super().__init__('Função eLU')
        self.alfa = alfa

    def __call__(self, x):
        return np.where(x <= 0, self.alfa * (np.exp(x) - 1), x)

    def derivada(self, x):
        y = self.__call__(x)
        return np.where(x <= 0, self.alfa + y, 1)


class FuncSoftmax(FuncAtivacaoGenerica):
    def __init__(self):
        super().__init__('Função Softmax')

    def __call__(self, x):
        expx = np.exp(x)
        return expx / np.sum(expx, axis=1, keepdims=True)

    @staticmethod
    def derivada(_x):
        return 1
