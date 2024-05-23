import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import collections
import random

TIME_STEP = 0.5
MAX_SPEED = 5
ACTION_BOUND = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

observe_dim = 4
observe_feature_dim = 180
num_layers = 1
locate_feature_dim = 100

la_feature_dim = 200  # locate & action


class Scenario:
    platform_x = 100
    platform_y = 100
    num_agents = 0

    def __init__(self, platform_x, platform_y, num_agents):
        self.num_agents = num_agents
        self.agents = []
        self.platform_x = platform_x
        self.platform_y = platform_y

        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.platform_x)
        self.ax.set_ylim(0, self.platform_y)
        self.ax.autoscale(enable=False)
        # self.ax.set_aspect('equal', adjustable='datalim')
        plt.xlim(0, self.platform_x)
        plt.ylim(0, self.platform_y)
        plt.grid(True)
        plt.ion()

        attempts = 0
        while len(self.agents) < self.num_agents and attempts < 1000:
            new_dot = Dot(self)
            if new_dot.collide() == False:
                self.agents.append(new_dot)
            attempts += 1

        for agent in self.agents:
            agent.add_neighbors()

        self.main_agent = self.agents[0]

    def reset(self):
        self.agents = []
        attempts = 0
        while len(self.agents) < self.num_agents and attempts < 1000:
            new_dot = Dot(self)
            if new_dot.collide() == False:
                self.agents.append(new_dot)
            attempts += 1

        for agent in self.agents:
            agent.add_neighbors()

        self.main_agent = self.agents[0]

    def step(self, main_action):
        pre_x = self.main_agent.x
        actions = [main_action]
        for agent in self.agents[1:]:  # 排除第0个的main_agent
            actions.append(agent.network.take_action(agent.state))
        for agent, a in zip(self.agents, actions):
            agent.v = agent.v + a * TIME_STEP
            agent.v = torch.clamp(agent.v, -MAX_SPEED, MAX_SPEED)
            agent.x = agent.x + agent.v * TIME_STEP
        for agent in self.agents:
            agent.add_neighbors()

        done = False

        if self.main_agent.collide():
            reward = -10000 * self.main_agent.collide()
            done = False

        elif torch.norm(self.main_agent.x - self.main_agent.goal) < self.main_agent.body_radius:
            # reward = 1000
            reward = 1000
            done = True

        else:
            reward1 = -torch.norm(self.main_agent.x - self.main_agent.goal).item()
            # print(f"距离奖励reward1:{reward1}")
            reward2 = 0
            for neighbor in self.main_agent.neighbors:
                distance = torch.norm(self.main_agent.x - neighbor.x).item()
                # reward2 += -1000 * (1 / (distance - 1.9 * self.main_agent.body_radius)
                # - 1 / (self.main_agent.detect_radius - 1.9 * self.main_agent.body_radius))
                if distance < self.main_agent.safe_radius:
                    # reward2 += -10000 * 1 / (distance - 0 * self.main_agent.body_radius)
                    reward2 -= 1000
            # print(f"奖碰撞励reward2:{reward2}")
            reward = reward1 + 0 * reward2
        return reward, done

    def render(self):
        self.ax.clear()  # 清除之前的绘图
        self.ax.set_xlim(-self.platform_x, 2 * self.platform_x)  # 重设坐标轴限制
        self.ax.set_ylim(-self.platform_y, 2 * self.platform_y)
        self.ax.set_aspect('equal', adjustable='datalim')
        plt.grid(True)

        for i, agent in enumerate(self.agents):
            x = agent.x[0].item()
            y = agent.x[1].item()
            detect_circle = plt.Circle((x, y), agent.detect_radius, color='steelblue', fill=False)
            safe_circle = plt.Circle((x, y), agent.safe_radius, color='pink', fill=False)
            self.ax.add_patch(detect_circle)
            self.ax.add_patch(safe_circle)
            self.ax.text(x, y, str(i), color='black', ha='center', va='center')  # 在圆心添加编号

            body_circle = plt.Circle((x, y), agent.body_radius, color='red', fill=True)
            self.ax.add_patch(body_circle)
            goal_x = agent.goal[0].item()
            goal_y = agent.goal[1].item()
            plt.scatter(goal_x, goal_y, color="crimson")
            plt.plot([goal_x, x], [goal_y, y], linestyle='--', color='peru')
        self.figure.canvas.draw()
        plt.pause(0.05)

    def render_gif(self,ax):
        ax.clear()  # 清除之前的绘图
        ax.set_xlim(-self.platform_x, 2 * self.platform_x)  # 重设坐标轴限制
        ax.set_ylim(-self.platform_y, 2 * self.platform_y)
        ax.set_aspect('equal', adjustable='datalim')
        plt.grid(True)

        for i, agent in enumerate(self.agents):
            x = agent.x[0].item()
            y = agent.x[1].item()
            detect_circle = plt.Circle((x, y), agent.detect_radius, color='steelblue', fill=False)
            safe_circle = plt.Circle((x, y), agent.safe_radius, color='pink', fill=False)
            ax.add_patch(detect_circle)
            ax.add_patch(safe_circle)
            ax.text(x, y, str(i), color='black', ha='center', va='center')  # 在圆心添加编号

            body_circle = plt.Circle((x, y), agent.body_radius, color='red', fill=True)
            ax.add_patch(body_circle)
            goal_x = agent.goal[0].item()
            goal_y = agent.goal[1].item()
            plt.scatter(goal_x, goal_y, color="crimson")
            plt.plot([goal_x, x], [goal_y, y], linestyle='--', color='peru')

class Dot:
    network = None

    def __init__(self, scenario):
        self.scenario = scenario
        self.body_radius = 5
        self.detect_radius = 50
        self.safe_radius = 12
        self.goal = torch.tensor([torch.rand(1).item() * self.scenario.platform_x,
                                  torch.rand(1).item() * self.scenario.platform_y],
                                 requires_grad=False).to(device)
        self.x = torch.tensor([torch.rand(1).item() * self.scenario.platform_x,  # 随机初始化位置x
                               torch.rand(1).item() * self.scenario.platform_y],
                              requires_grad=False).to(device)
        self.v = torch.tensor([torch.rand(1).item() * MAX_SPEED,  # 随机初始化位置x
                               torch.rand(1).item() * MAX_SPEED],
                              requires_grad=False).to(device)
        self.neighbors = set()

        # self.add_neighbors()

    @property
    def xv(self):
        return torch.cat((self.x, self.v), dim=0).detach().requires_grad_(False)

    @property
    def locate(self):
        return torch.cat((self.goal - self.x, self.v), dim=0).detach().requires_grad_(False)

    @property
    def observe(self):
        xv_list = [neighbor.xv - self.xv for neighbor in self.neighbors]
        # 按危险性从低到高，对neighbors进行排序
        xv_list = sorted(xv_list,
                         key=lambda xv: (torch.dot(xv[2:], xv[:2]).item() * abs(torch.dot(xv[2:], xv[:2]).item())
                                         + 2 * self.network.action_bound * torch.norm(xv[:2]).item()),
                         reverse=True)  # 从大到小
        xv_list.insert(0, torch.tensor([0, 0, 0, 0], dtype=torch.float).to(device))  # 第一行插入0防止为空
        return torch.stack(xv_list, dim=0).detach().requires_grad_(False).to(device)

    @property
    def state(self):
        return torch.cat((self.locate.unsqueeze(0), self.observe), dim=0).detach().requires_grad_(False)

    def collide(self):  # 返回是否发生碰撞
        n = 0
        for agent in self.scenario.agents:
            if agent is not self:
                if (torch.norm(self.x - agent.x) < 2 * self.body_radius or
                        torch.norm(self.goal - agent.goal) < 2 * self.body_radius):
                    n += 1
        return n

    def add_neighbors(self):
        self.neighbors = set()  # 清空原来的neighbors
        for agent in self.scenario.agents:
            if (agent is not self) and (torch.norm(self.x - agent.x) < self.detect_radius):
                self.neighbors.add(agent)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=observe_dim,
                                  hidden_size=observe_feature_dim,
                                  num_layers=num_layers,
                                  batch_first=True)
        self.fc_locate = torch.nn.Linear(state_dim, locate_feature_dim)
        hidden_dim = locate_feature_dim + observe_feature_dim
        # hidden_dim = locate_feature_dim

        self.fc_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = ACTION_BOUND  # action_bound是环境可以接受的动作最大值

    def forward(self, x):  # x的形状为(batch_size, sequence_length, state_dim)构成的PackedSequence
        x, lengths = pad_packed_sequence(x, batch_first=True)
        locate = x[:, 0, :]  # locate的形状为(batch_size, state_dim)
        observe = x[:, 1:, :]
        observe = pack_padded_sequence(observe, lengths - 1, batch_first=True, enforce_sorted=False)

        locate_feature = F.relu(self.fc_locate(locate))
        output, (observe_feature, cell) = self.lstm(observe)
        observe_feature = (observe_feature[-1])  # hidden的形状是(n*num_layers, batch_size, hidden_size)
        feature = torch.cat((locate_feature, observe_feature), dim=1)
        # feature = locate_feature
        feature = F.relu(self.fc_1(feature))
        a = torch.tanh(self.fc_2(feature)) * self.action_bound

        # a = self.action_bound * locate[:, :2] / torch.norm(locate[:, :2], dim=1)
        return a
        # return self.fc_2(feature)


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QValueNet, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=observe_dim,
                                  hidden_size=observe_feature_dim,
                                  num_layers=num_layers,
                                  batch_first=True)
        self.fc_la = torch.nn.Linear(state_dim + action_dim, la_feature_dim)
        hidden_dim = la_feature_dim + observe_feature_dim
        # hidden_dim = la_feature_dim

        self.fc_1 = torch.nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc_2 = torch.nn.Linear(int(hidden_dim / 2), 1)

    def forward(self, x, a):  # x的形状为(batch_size, sequence_length, state_dim)构成的PackedSequence
        action = a
        x, lengths = pad_packed_sequence(x, batch_first=True)
        locate = x[:, 0, :]
        observe = x[:, 1:, :]
        observe = pack_padded_sequence(observe, lengths - 1, batch_first=True, enforce_sorted=False)
        la = torch.cat([locate, action], dim=1)
        la_feature = F.relu(self.fc_la(la))
        output, (observe_feature, cell) = self.lstm(observe)
        observe_feature = (observe_feature[-1])  # hidden的形状是(n*num_layers, batch_size, hidden_size)

        feature = torch.cat((la_feature, observe_feature), dim=1)
        # feature = la_feature
        feature = F.relu(self.fc_1(feature))
        q = self.fc_2(feature)
        return q


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, action_dim, sigma=0,
                 actor_lr=0.001, critic_lr=0.001, tau=None, gamma=None, device=None, explore=False):
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.explore = explore  # 在训练时应当设置为True
        self.action_bound = ACTION_BOUND

    def take_action(self, state):
        state = pack_padded_sequence(state.unsqueeze(0), torch.tensor([state.size(0)]), batch_first=True,
                                     enforce_sorted=False)
        action = self.actor(state).squeeze(0)
        if self.explore:
            # 给动作添加噪声，增加探索
            action = action + self.sigma * torch.randn(self.action_dim).to(self.device)

        return action

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

    def update(self, transition_dict):
        states = transition_dict['states']
        state_lengths = [x.size(0) for x in states]
        state_lengths = torch.tensor(state_lengths).requires_grad_(False)
        states = pad_sequence(states, batch_first=True).detach().requires_grad_(False).to(self.device)
        states = pack_padded_sequence(states, state_lengths, batch_first=True, enforce_sorted=False)
        # states = torch.stack(transition_dict['states'], dim=0).detach().requires_grad_(True).to(self.device)

        actions = torch.stack(transition_dict['actions'], dim=0).detach().requires_grad_(True).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).detach().to(self.device)

        next_states = transition_dict['next_states']
        next_state_lengths = [len(x) for x in next_states]
        next_state_lengths = torch.tensor(next_state_lengths).requires_grad_(False)
        next_states = pad_sequence(next_states, batch_first=True).detach().requires_grad_(False).to(self.device)
        next_states = pack_padded_sequence(next_states, next_state_lengths, batch_first=True, enforce_sorted=False)
        # next_states = torch.stack(transition_dict['next_states'], dim=0).detach().requires_grad_(True).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).detach().to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states)).detach().requires_grad_(False)
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=False)
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=False)
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.target_critic, self.tau)
        self.soft_update(self.actor, self.target_actor, self.tau)

    def load(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def evaluate(self):
        self.explore = False
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()
        self.target_actor.eval()

    def train(self):
        self.explore = True
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)
