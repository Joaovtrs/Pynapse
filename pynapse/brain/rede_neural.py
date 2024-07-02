class RedeNeural:
    def __init__(self, camadas):
        self.camadas = camadas

    def __str__(self):
        resp = ''
        for i, camada in enumerate(self.camadas):
            resp += f'Camada {i + 1}:\n' + str(camada) + '\n\n'
        return resp

    def __repr__(self):
        return self.__str__()

    def __call__(self, x, detalhes=False):
        for i, camada in enumerate(self.camadas):
            x = camada(x)
            if detalhes:
                print(f'Resultado da camadab {i + 1}:\n{x}')
        return x

    def backprop(self, x, y, func_custo, lr):
        y_pred = self(x)

        dout = [func_custo.derivada(y, y_pred)]

        for camada in self.camadas[::-1]:
            dout.append(camada.backprop(dout[-1]))

        for camada in self.camadas:
            camada.atualizar(lr)

        return func_custo(y, y_pred)
