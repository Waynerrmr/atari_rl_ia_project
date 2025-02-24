# w: usem o format on save do VSCode pra manter o código bonito
# w: warnings nas imports? acontece, só bora
# w: prazo é amanhã, sem tempo irmão, adiantem

import gym
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from agents.dqn_agent import RedeDQN
from agents.memory import ReplayMemory

# Configuração do ambiente
NOME_AMBIENTE = "Pong-ramNoFrameskip-v4"
# Hiperparâmetros
NUM_EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
EXPLORATION = 0.60
EXPLORATION_DECAY = 0.999
MIN_EXPLORATION = 0.01

ambiente = gym.make(NOME_AMBIENTE)
formato_entrada = (1, 84, 84)
num_acoes = ambiente.action_space.n

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = RedeDQN(formato_entrada, num_acoes).to(dispositivo)
otimizador = optim.Adam(modelo.parameters(), lr=1e-4)
memoria = ReplayMemory(10000)

def preprocessar(estado):
    estado = np.mean(estado, axis=-1)
    estado = np.resize(estado, (84, 84))
    return np.array(estado, dtype=np.float32) / 255.0

for episodio in range(NUM_EPISODES):
    estado = preprocessar(ambiente.reset()[0])
    recompensa_total = 0

    for _ in range(2000):
        if np.random.rand() < EXPLORATION:
            acao = ambiente.action_space.sample()
        else:
            with torch.no_grad():
                tensor_estado = torch.tensor(estado, dtype=torch.float32, device=dispositivo).unsqueeze(0).unsqueeze(1)
                valores_q = modelo(tensor_estado)
                acao = torch.argmax(valores_q).item()

        proximo_estado, recompensa, terminou, _, _ = ambiente.step(acao)
        proximo_estado = preprocessar(proximo_estado)
        memoria.armazenar(estado, acao, recompensa, proximo_estado, terminou)
        estado = proximo_estado
        recompensa_total += recompensa

        if len(memoria) > BATCH_SIZE:
            batch = memoria.amostrar(BATCH_SIZE)
            estados, acoes, recompensas, proximos_estados, finais = zip(*batch)

            estados = torch.tensor(np.array(estados), dtype=torch.float32, device=dispositivo).unsqueeze(1)
            acoes = torch.tensor(acoes, dtype=torch.int64, device=dispositivo).unsqueeze(-1)
            recompensas = torch.tensor(recompensas, dtype=torch.float32, device=dispositivo)
            proximos_estados = torch.tensor(np.array(proximos_estados), dtype=torch.float32, device=dispositivo).unsqueeze(1)
            finais = torch.tensor(finais, dtype=torch.float32, device=dispositivo)

            valores_q = modelo(estados).gather(1, acoes)
            proximos_q = modelo(proximos_estados).max(1)[0].detach()
            alvo_q = recompensas + GAMMA * proximos_q * (1 - finais)

            perda = F.mse_loss(valores_q, alvo_q.unsqueeze(1))
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()

        if terminou: 
            break

    EXPLORATION = max(MIN_EXPLORATION, EXPLORATION * EXPLORATION_DECAY)
    if (episodio + 1) % 10 == 0:
        torch.save(modelo.state_dict(), f"models/dqn_{NOME_AMBIENTE}_ep{episodio + 1}.pth")
        print(f"Modelo salvo após {episodio + 1} episódios")
    print(f"Episódio {episodio + 1}, Recompensa: {recompensa_total:.2f}, Exploração: {EXPLORATION:.2f}")

torch.save(modelo.state_dict(), f"models/dqn_{NOME_AMBIENTE}_{NUM_EPISODES}_explr{EXPLORATION}.pth")
print(f"Modelo final salvo em models/dqn_{NOME_AMBIENTE}_{NUM_EPISODES}_explr{EXPLORATION}.pth")
ambiente.close()
