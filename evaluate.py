import torch
from utils import Scenario, Dot, DDPG
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from datetime import datetime
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = 4
action_dim = 2
gamma = 0.98  # return的缩减

platform_x = 100
platform_y = 100
num_agents = 5  # 智能体个数

seed = int(time.time())
torch.manual_seed(seed)


def show():  # 用于动态显示运动
    scenario = Scenario(platform_x, platform_y, num_agents)
    agent = scenario.main_agent

    network = DDPG(state_dim, action_dim, device=device, explore=False)
    network.load(actor_path="model/num5actor.pth", critic_path="model/num5critic.pth")
    network.evaluate()
    Dot.network = network
    episode_return = 0
    done = False
    steps = 0
    while steps < 300 and not done:
        steps += 1
        state = agent.state
        action = agent.network.take_action(state)
        reward, done = scenario.step(action)
        scenario.render()

        # episode_return += reward
        episode_return = episode_return * gamma + reward


def update(frame, ax, agent, scenario):  # 用于生成动图的每一帧
    state = agent.state
    action = agent.network.take_action(state)
    reward, done = scenario.step(action)
    scenario.render_gif(ax)


def save_gif():  # 用于生成动图
    scenario = Scenario(platform_x, platform_y, num_agents)
    agent = scenario.main_agent
    network = DDPG(state_dim, action_dim, device=device, explore=False)
    network.load(actor_path="model/actor.pth", critic_path="model/critic.pth")
    network.evaluate()
    Dot.network = network
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, update, fargs=(ax, agent, scenario), frames=range(100), interval=50)
    current_time = datetime.now().strftime("%m_%d_%H%M%S")
    file_name = f"gif/{current_time}.gif"
    ani.save(file_name, writer='Pillow')


if __name__ == "__main__":
    # show()
    for i in range(5):  # 一次生成多张动图
        save_gif()  # 可以使用show()或者save_gif
