# Teste inicial do ambiente Atari
import gym

ambiente = gym.make("PongNoFrameskip-v4", render_mode="human")
ambiente.reset()

for _ in range(1000):
    ambiente.render()
    acao = ambiente.action_space.sample()
    ambiente.step(acao)

ambiente.close()