import random
import collections
import numpy as np

class ReplayMemory:
    def __init__(self, capacidade):
        self.memoria = collections.deque(maxlen=capacidade)

    def armazenar(self, estado, acao, recompensa, prox_estado, terminou):
        self.memoria.append((estado, acao, recompensa, prox_estado, terminou))

    def amostrar(self, tamanho_lote):
        return random.sample(self.memoria, tamanho_lote)

    def __len__(self):
        return len(self.memoria)