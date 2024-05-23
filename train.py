import torch
import matplotlib.pyplot as plt
import random
from utils import Scenario, Dot, ReplayBuffer, DDPG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 300
state_dim = 4
action_dim = 2  # 控制x和y方向的加速度

gamma = 0.98  # return的缩减
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 70
batch_size = 64  # 64
sigma = 0.5  # 高斯噪声标准差

platform_x = 100
platform_y = 100
num_agents = 2  # 智能体个数

random.seed(0)
torch.manual_seed(0)

network = DDPG(state_dim, action_dim, sigma,
               actor_lr, critic_lr, tau, gamma, device)
# network.load(actor_path="model/actor.pth", critic_path="model/critic.pth")
network.train()
Dot.network = network

scenario = Scenario(platform_x, platform_y, num_agents)
agent = scenario.main_agent

replay_buffer = ReplayBuffer(buffer_size)

return_list = []
for episode in range(num_episodes):

    episode_return = 0
    done = False
    scenario.reset()
    agent = scenario.main_agent

    steps = 0
    while steps < 200 and not done:
        steps += 1
        state = agent.state
        action = agent.network.take_action(state)
        reward, done = scenario.step(action)
        # scenario.render()

        next_state = agent.state
        replay_buffer.add(state, action, reward, next_state, done)
        # episode_return += reward
        episode_return = episode_return * gamma + reward

        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            agent.network.update(transition_dict)

    return_list.append(episode_return)
    if done:
        print("reached")
    print(f"Episode:{episode}, Reward:{episode_return}")

plt.figure(2)
plt.plot(return_list, marker='o')
plt.xlabel("episode")
plt.ylabel("return")
plt.savefig('step100_.png')
plt.show()

torch.save(Dot.network.actor.state_dict(), "model/actor.pth")
torch.save(Dot.network.critic.state_dict(), "model/critic.pth")
