import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 計算 Conv 層輸出的大小: 32 channel * 8 * 8
        conv_out_size = 32 * input_shape[1] * input_shape[2]
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class ReplayBuffer:
    # FIFO
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 兩個網路：Policy Net (更新快), Target Net (更新慢，為了穩定)
        self.policy_net = DQN((self.cfg.CARGO_TYPES+1, self.cfg.MAP_SIZE, self.cfg.MAP_SIZE), 4).to(self.device)
        self.target_net = DQN((self.cfg.CARGO_TYPES+1, self.cfg.MAP_SIZE, self.cfg.MAP_SIZE), 4).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 同步權重
        self.target_net.eval() # Target Net 不需訓練
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.LR)
        self.memory = ReplayBuffer(self.cfg.MEMORY_CAPACITY)
        self.epsilon = self.cfg.EPSILON_START

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.cfg.BATCH_SIZE:
            return

        # 1. 取樣
        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.BATCH_SIZE)
        
        # 轉為 Tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 2. 計算 Current Q (機器人猜的)
        # gather: 根據 action index 抓出對應的 Q 值
        current_q = self.policy_net(states).gather(1, actions)

        # 3. 計算 Target Q (正確答案)
        # 改double DQN (避免過度樂觀)
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions)
        
        expected_q = rewards + (self.cfg.GAMMA * next_q * (1 - dones))

        # 4. 計算 Loss (MSE)
        loss = nn.MSELoss()(current_q, expected_q)

        # 5. 反向傳播 (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 限制成長幅度(有時候會爆衝到超過理論值)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def update_epsilon(self):
        self.epsilon = max(self.cfg.EPSILON_END, self.epsilon * self.cfg.EPSILON_DECAY)
