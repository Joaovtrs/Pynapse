import numpy as np


class RedeNeural:
    def __init__(self, camadas):
        self.camadas = camadas

    def __str__(self):
        resp = '\nRede:\n'

        for i, camada in enumerate(self.camadas):
            resp += f'\tCamada {i + 1}:\n' + str(camada) + '\n\n'

        n_parametros = sum(
            [camada.get_paran_cout() for camada in self.camadas]
        )

        resp += f'\tTotal de parêmatros: {n_parametros}\n\n'

        return resp

    def __repr__(self):
        return self.__str__()

    def __call__(self, x, dropout_prob=0, detalhes=False):
        for i, camada in enumerate(self.camadas):
            x = camada(x, dropout_prob=dropout_prob)
            if detalhes:
                print(f'Resultado da camada {i + 1}:\n{x}')
        return x

    def backprop(self, x, y, func_custo, lr, momentum=0, dropout_prob=0):
        y_pred = self(x, dropout_prob=dropout_prob)

        dout = [func_custo.derivada(y, y_pred)]

        for camada in self.camadas[::-1]:
            dout.append(camada.backprop(dout[-1]))

        for camada in self.camadas[::-1]:
            camada.atualizar(lr, momentum)

        custo_reg = np.sum(
            [
                camada.func_regularizacao(camada.pesos, x.shape[0])
                for camada in self.camadas
            ]
        )
        return func_custo(y, y_pred) + custo_reg
