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
        y_pred = [x]

        for camada in self.camadas:
            y_pred.append(camada(y_pred[-1]))

        dout = [func_custo.derivada(y, y_pred[-1])]

        for i, camada in enumerate(self.camadas[::-1]):
            dout.append(camada.backprop(y_pred[-(i + 2)], dout[-1], lr))

        return func_custo(y, y_pred[-1])
