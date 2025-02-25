# Projeto: Aprendizado por Reforço no Pong-ramNoFrameskip-v4

## Integrantes
- WAYNER RORAN MORAES ROLIM / 202100023048
- JOSÉ ANTONIO DE GOES NETO / 202100022828
- ALMIR VINÍCIUS BISPO DO NASCIMENTO / 202100011181

## 📌 Visão Geral
Este projeto implementa um agente de aprendizado por reforço para jogar **Pong-ramNoFrameskip-v4** utilizando **Deep Q-Network (DQN)**.

## 📂 Estrutura do Projeto
```
atari_rl/
│── src/
│   ├── train.py       # Script de treinamento do agente
│   ├── test.py        # Script para testar o agente treinado
│   ├── test_env.py    # Teste básico do ambiente
│   ├── agents/
│   │   ├── dqn_agent.py  # Definição do modelo DQN
│   │   ├── memory.py     # Implementação do Replay Memory
│── models/           # Diretório onde os modelos treinados serão salvos
│── requirements.txt  # Lista de dependências do projeto
│── README.md         # Documentação do projeto
```

## 🛠️ Dependências
As dependências do projeto estão listadas no arquivo `requirements.txt`.

### 📥 Instalando as dependências
Execute o comando abaixo para instalar todas as bibliotecas necessárias:
```sh
pip install -r requirements.txt
```
Obs.: Foi utilizado o [python 3.10](https://www.python.org/downloads/release/python-3100/), pois haviam mais informações dessa versão


## 🚀 Como Executar

### 🔹 1. Testar o ambiente
Antes de treinar o agente, verifique se o ambiente está configurado corretamente:
```sh
python src/test_env.py
```

### 🔹 2. Treinar o agente
Para iniciar o treinamento do agente, execute:
```sh
python src/train.py
```
Isso treinará o modelo e salvará os pesos no diretório `models/`.

### 🔹 3. Testar o modelo treinado
Após o treinamento, execute o teste para avaliar o desempenho do agente:
```sh
python src/test.py
```

## 📦 Arquivo `requirements.txt`
O arquivo `requirements.txt` contém todas as bibliotecas necessárias para rodar o projeto. Aqui está o conteúdo:
```
torch
numpy
gym[atari]
stable-baselines3
```

Para exportar suas dependências para outro dispositivo, use o comando:
Sugiro que não instalem mais nada, já está rodando normalmente, não vamos estragar na hora de enviar o projeto
```sh
pip freeze > requirements.txt
```

## 📄 Relatório/Artigo Técnico-Científico
- [Versão em português](https://onedrive.live.com/:w:/g/personal/EC40BC6C7F2E6E05/EbsqaJqrZahIt5rrkTG2hPwBeHQLg2ztW6ODOj-MfOW5ZA?resid=EC40BC6C7F2E6E05!s9a682abb65ab48a8b79aeb9131b684fc&ithint=file%2Cdocx&e=aXT2Vm&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3cvYy9lYzQwYmM2YzdmMmU2ZTA1L0Vic3FhSnFyWmFoSXQ1cnJrVEcyaFB3QmVIUUxnMnp0VzZPRE9qLU1mT1c1WkE_ZT1hWFQyVm0)
- [Versão em inglês](https://onedrive.live.com/:w:/g/personal/EC40BC6C7F2E6E05/EcI5bV1G381Hlbse1b0qgZQBXCAtuYOwWXhjmYg4rfb-EQ?resid=EC40BC6C7F2E6E05!s5d6d39c2df4647cd95bb1ed5bd2a8194&ithint=file%2Cdocx&e=CJYHa5&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3cvYy9lYzQwYmM2YzdmMmU2ZTA1L0VjSTViVjFHMzgxSGxic2UxYjBxZ1pRQlhDQXR1WU93V1hoam1ZZzRyZmItRVE_ZT1DSllIYTU)
