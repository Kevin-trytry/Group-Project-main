import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

import robot as rb  # 匯入遊戲邏輯
from config import Config

# 註冊環境 ID
register(
    id='cargo-env-v0',
    entry_point='cargo_env:CargoEnv',
)

class CargoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps': 4}

    def __init__(self, render_mode=None):
        self.cfg = Config()
        self.render_mode = render_mode
        self.map_size = self.cfg.MAP_SIZE
        
        # 實例化遊戲核心 (Logic)
        # render_on 決定是否開啟 pygame 視窗
        self.game = rb.CargoGame(config=self.cfg, render_on=(render_mode == 'human'))
        
        # 定義 Action Space (0, 1, 2, 3)
        self.action_space = spaces.Discrete(len(rb.RobotAction))
        
        # 定義 Observation Space (原本 environment.py 的矩陣形狀)
        # Shape: (4, MAP_SIZE, MAP_SIZE)
        self.observation_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(4, self.map_size, self.map_size),
            dtype=np.float32
        )

    def _get_observation(self):
        obs = np.zeros((4, self.map_size, self.map_size), dtype=np.float32)
        
        # Layer 0: Robot
        rx, ry = self.game.robot_pos
        obs[0, rx, ry] = 1.0
        
        # Layer 1-3: Cargos (limit, good, bad)     
        for i in range(self.cfg.CARGO_TYPES):
            for item in self.game.all_cargos[i]:
                if i == 0:
                    obs[1, item.row, item.col] = self.game.all_cargos[0].remain_lifetime / self.game.all_cargos[0].lifetime
                elif item.acitve:
                    obs[i+1, item.row, item.col] = 1.0
            
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 呼叫遊戲核心的 reset
        # 如果需要 seed，可以在 CargoGame 的 reset 實作 seed 參數
        self.game.reset()
        
        if self.render_mode == 'human':
            self.game.render()
            
        return self._get_observation(), {}

    def step(self, action):
        # 1. 將整數動作轉換為 Enum
        robot_action = rb.RobotAction(action)
        
        # 2. 讓遊戲核心執行動作，只拿回 reward 和 done
        reward, done = self.game.perform_action(robot_action)
        
        # 3. 重新計算觀察值 (Observation)
        obs = self._get_observation()
        
        # 4. 處理 Render
        if self.render_mode == 'human':
            self.game.render()
            
        info = {}
        
        # 回傳標準 Gym 格式
        return obs, reward, done, False, info

    def render(self):
        self.game.render()

# 測試用
if __name__ == "__main__":
    env = gym.make('cargo-env-v0', render_mode='human')
    obs, _ = env.reset()
    print("Env Reset done. Obs shape:", obs.shape)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            env.reset()
