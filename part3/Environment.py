import gymnasium as gym
import numpy as np
import random
from cargo import GoodCargo, BadCargo, LimitedCargo


class Config:
    def __init__(self):
        self.MAP_SIZE = 8
        
        self.num_limit = 1  # 限時包裹的數量
        self.num_good = 3   # 一般包裹的數量
        self.num_bad = 2    # 負分包裹的數量
        self.total_cargo = self.num_limit + self.num_good + self.num_bad    
        
        self.STEP_COST = -1      # 每走一步的消耗 (鼓勵最短路徑)

class Envirnment:
    def __init__(self, config):
        self.cfg = config
        self.state_dim = (4, self.cfg.MAP_SIZE, self.cfg.MAP_SIZE) # 4 Channels
        self.action_space = 4 # 上, 下, 左, 右
        self.reset()
    
    def reset(self):
        self.score = 0
        self.steps = 0
        self.map_size = self.cfg.MAP_SIZE
        
        positions = random.sample(range(self.map_size * self.map_size), self.cfg.total_cargo) 
        # 生成5個座標，機器人、高分限時包裹、一般包裹*3、負分包裹*2
        self.robot_pos = self._to_coord(positions[0])
        self.limit_cargo = LimitedCargo(positions[1])
        self.good_cargo = [GoodCargo(positions[2]), GoodCargo(positions[3]), GoodCargo(positions[4])]
        self.bad_cargo = [BadCargo(positions[5]), BadCargo(positions[6])]
        
        return self._get_observation()
    
    def _to_coord(self, idx):
        return [idx // self.map_size, idx % self.map_size]
    
    def _get_observation(self):
        """將狀態轉為神經網路看得懂的 Tensor (Channels, H, W)"""
        # 每一種物件獨立一層Channel
        # Channel 0: 機器人位置
        # Channel 1: 高分包裹 (數值為剩餘時間比例)
        # Channel 2: 低分包裹
        # Channel 3: 負分包裹
        obs = np.zeros((4, self.map_size, self.map_size), dtype=np.float32)
        
        # Ch 0
        rx, ry = self.robot_pos
        obs[0, rx, ry] = 1.0
        
        # Ch 1 (High Pkg with Time decay info)
        if self.limit_cargo.active:
            lx, ly = self.limit_cargo.pos
            obs[1, lx, ly] = self.limit_cargo.remain_lifetime / self.limit_cargo.lifetime
            
        # Ch 2 (Low Pkg)
        if self.good_cargo.active:
            for gx, gy in self.good_cargo.pos:
                obs[2, gx, gy] = 1.0
            
        # Ch 3 (Neg Pkg)
        if self.bad_cargo.active:
            for bx, by in self.bad_cargo.pos:
                obs[3, bx, by] = 1.0
            
        return obs
    
    def step(self, action):
        """執行動作: 0:上, 1:下, 2:左, 3:右"""
        self.steps += 1
        x, y = self.robot_pos
        
        # 移動邏輯
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1
        
        reward = self.cfg.STEP_COST
        done = False
        
        # 邊界檢查 (撞牆)
        if x < 0 or x >= self.map_size or y < 0 or y >= self.map_size:
            reward += -5 # 撞牆懲罰
            x, y = self.robot_pos # 彈回原位
        else:
            self.robot_pos = [x, y]

        # 更新高分包裹
        self.limit_cargo.update()
        if self.limit_cargo.active:
            if self.robot_pos == self.limit_cargo.pos:
                reward += self.limit_cargo.get_reward()
        
        # 檢查低分包裹
        for pkg in self.good_cargo:
            if pkg.active and self.robot_pos == pkg.pos:
                reward += pkg.get_reward()
            
        # 檢查負分包裹
        for pkg in self.bad_cargo:
            if pkg.active and self.robot_pos == pkg.pos:
                reward += pkg.get_reward()
            
        # 結束條件: 所有正分包裹都拿完 或 步數過多
        for pkg in self.good_cargo:
            if pkg.active:
              done = False
              
        if done and not self.limit_cargo.active:
           reward += 20  
            
        if self.steps >= self.cfg.MAX_STEPS:
            done = True
            
        self.score += reward
        return self._get_observation(), reward, done
    
 