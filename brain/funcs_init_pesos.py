import numpy as np


class FuncInitPesosGenerica:
    def __init__(self, nome):
        self.nome = nome

    def __str__(self):
        return self.nome

    def __repr__(self):
        return self.__str__()


class FuncZeros(FuncInitPesosGenerica):
    def __init__(self):
        super().__init__('Função Inicialização de Pesos com Zeros')

    def __call__(self, linhas, colunas):
        return np.zeros((linhas, colunas))


class FuncUns(FuncInitPesosGenerica):
    def __init__(self):
        super().__init__('Função Inicialização de Pesos com Uns')

    def __call__(self, linhas, colunas):
        return np.ones((linhas, colunas))


class FuncRandomNormal(FuncInitPesosGenerica):
    def __init__(self):
        super().__init__('Função Inicialização de Pesos Random Normal')

    def __call__(self, linhas, colunas):
        return np.random.randn(linhas, colunas)


class FuncRandomUniforme(FuncInitPesosGenerica):
    def __init__(self):
        super().__init__('Função Inicialização de Pesos Random Uniforme')

    def __call__(self, linhas, colunas):
        return np.random.rand(linhas, colunas)


class FuncGlorotNormal(FuncInitPesosGenerica):
    def __init__(self):
        super().__init__('Função Inicialização de Pesos Glorot Normal')

    def __call__(self, linhas, colunas):
        desvio = np.sqrt(2 / (linhas + colunas))
        return desvio * np.random.randn(linhas, colunas)


class FuncGlorotUniforme(FuncInitPesosGenerica):
    def __init__(self):
        super().__init__('Função Inicialização de Pesos Glorot Uniforme')

    def __call__(self, linhas, colunas):
        desvio = np.sqrt(6 / (linhas + colunas))
        return 2 * desvio * np.random.rand(linhas, colunas) - desvio
