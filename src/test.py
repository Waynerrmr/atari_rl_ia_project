import os
import gym
import torch
import numpy as np
from agents.dqn_agent import RedeDQN

# Configuração do ambiente
NOME_AMBIENTE = "Pong-ramNoFrameskip-v4"
ambiente = gym.make(NOME_AMBIENTE, render_mode="human")
formato_entrada = (1, 84, 84)
num_acoes = ambiente.action_space.n

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = RedeDQN(formato_entrada, num_acoes).to(dispositivo)

# Localiza o modelo mais recente salvo
diretorio_modelos = "models/"
arquivos_modelo = [f for f in os.listdir(diretorio_modelos) if f.startswith(f"dqn_{NOME_AMBIENTE}_ep") and f.endswith(".pth")]

if arquivos_modelo:
    modelo_recente = sorted(arquivos_modelo, key=lambda x: int(x.split("_ep")[1].split(".pth")[0]))[-1]
    # caminho_modelo = 'models/dqn_Pong-ramNoFrameskip-v4_ep30.pth'
    caminho_modelo = os.path.join(diretorio_modelos, modelo_recente)
    print(f"✅ Carregando modelo mais recente: {caminho_modelo}")
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=dispositivo))
else:
    print("❌ Nenhum modelo encontrado. Verifique se há modelos salvos na pasta 'models/'.")
    exit()

modelo.eval()

def preprocessar(estado):
    estado = np.mean(estado, axis=-1)
    estado = np.resize(estado, (84, 84))
    return np.array(estado, dtype=np.float32) / 255.0

estado = preprocessar(ambiente.reset()[0])
total_recompensa = 0

for _ in range(1000):
    with torch.no_grad():
        tensor_estado = torch.tensor(estado, dtype=torch.float32, device=dispositivo).unsqueeze(0).unsqueeze(1)
        valores_q = modelo(tensor_estado)
        acao = torch.argmax(valores_q).item()

    proximo_estado, recompensa, terminou, _, _ = ambiente.step(acao)
    proximo_estado = preprocessar(proximo_estado)
    estado = proximo_estado
    total_recompensa += recompensa

    if terminou:
        break

print(f"🎯 Recompensa Total no Teste: {total_recompensa}")
ambiente.close()
