import torch
import numpy as np
import torch.nn as nn

class RedeDQN(nn.Module):
    def __init__(self, formato_entrada, num_acoes):
        super(RedeDQN, self).__init__()
        self.camada_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )

        tamanho_saida = self._calcular_saida_conv(formato_entrada)
        self.camada_fc = nn.Sequential(
            nn.Linear(tamanho_saida, 512),
            nn.ReLU(),
            nn.Linear(512, num_acoes)
        )

    def _calcular_saida_conv(self, formato):
        with torch.no_grad():
            saida = self.camada_conv(torch.zeros(1, *formato))
        return int(np.prod(saida.size()[1:]))

    def forward(self, x):
        x = x.float()
        x = self.camada_conv(x).view(x.size(0), -1)
        return self.camada_fc(x)
