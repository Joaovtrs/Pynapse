from pynapse.brain import Camada


def test_criacao_camada():
    camada = Camada(1, 1)

    assert isinstance(camada, Camada)


def test_tamanho_array_pesos():
    camada_1 = Camada(1, 1)
    camada_2 = Camada(3, 1)
    camada_3 = Camada(3, 4)

    assert camada_1.pesos.shape == (1, 1)
    assert camada_2.pesos.shape == (1, 3)
    assert camada_3.pesos.shape == (4, 3)


def test_tamanho_array_bias():
    camada_1 = Camada(1, 1)
    camada_2 = Camada(3, 1)
    camada_3 = Camada(3, 4)

    assert camada_1.bias.shape == (1, 1)
    assert camada_2.bias.shape == (1, 1)
    assert camada_3.bias.shape == (1, 4)
