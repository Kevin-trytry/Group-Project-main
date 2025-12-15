import pygame
import random
import sys
from os import path
from enum import Enum

from config import Config
from cargo import GoodCargo, BadCargo, LimitedCargo

class RobotAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class CargoGame:
    def __init__(self, config=None, render_on=False):
        self.cfg = config if config else Config()
        self.map_size = self.cfg.MAP_SIZE
        self.render_on = render_on
        
        # 遊戲狀態初始化
        self.robot_pos = (0, 0)
        self.steps = 0
        self.score = 0
        self.limit_cargo = []
        self.good_cargo = []
        self.bad_cargo = []
        self.all_cargos = []

        # 如果需要畫面顯示才初始化 Pygame
        self.window_surface = None
        if self.render_on:
            self._init_pygame()
        
        # 初始化遊戲狀態
        self.reset()

    def reset(self):
        self.steps = 0
        self.score = 0
        
        # 隨機生成位置
        total_cargo = self.cfg.NUM_LIMIT + self.cfg.NUM_GOOD + self.cfg.NUM_BAD
        positions = random.sample(range(self.map_size * self.map_size), total_cargo+1) 
        coordinates = [self._to_coord(pos) for pos in positions]
        
        start = 0
        end = self.cfg.NUM_LIMIT
        self.limit_cargo = [LimitedCargo(*c) for c in coordinates[start : end]]
        
        start = self.cfg.NUM_LIMIT
        end = self.cfg.NUM_GOOD + self.cfg.NUM_LIMIT
        self.good_cargo = [GoodCargo(*c) for c in coordinates[start : end]]
        
        start = self.cfg.NUM_LIMIT + self.cfg.NUM_GOOD
        end = self.cfg.NUM_LIMIT + self.cfg.NUM_GOOD + self.cfg.NUM_BAD
        self.bad_cargo = [BadCargo(*c) for c in coordinates[start : end]]

        self.all_cargos = [self.limit_cargo, self.good_cargo, self.bad_cargo] 
        
        self.robot_pos = tuple(coordinates[total_cargo])
        
    def _to_coord(self, idx):
        return (idx // self.map_size, idx % self.map_size)

    def perform_action(self, action: RobotAction):
        self.steps += 1
        rx, ry = self.robot_pos
        x, y = rx, ry
        
        # 解析動作
        if action == RobotAction.UP: x -= 1
        elif action == RobotAction.DOWN: x += 1
        elif action == RobotAction.LEFT: y -= 1
        elif action == RobotAction.RIGHT: y += 1
        
        reward = self.cfg.STEP_COST
        
        # 邊界檢查 (撞牆)
        if x < 0 or x >= self.map_size or y < 0 or y >= self.map_size:
            reward += -1 
            # 撞牆後位置維持原狀 (rx, ry)
        else:
            self.robot_pos = (x, y) # 更新位置

        # 更新限時包裹狀態
        for pkg in self.all_cargos[0]:
            pkg.update()
            
        # 檢查各類包裹
        for i in range(self.cfg.CARGO_TYPES):
            for pkg in self.all_cargos[i]:
                if pkg.active and self.robot_pos == pkg.pos:
                    reward += pkg.get_reward()
                    
        # 額外增加靠近限時包裹的reward，若已經沒有限時包裹了，額外增加靠近最近的包裹的reward
        targets = [p for p in self.limit_cargo if p.active]
        if not targets:
            targets = [p for p in self.good_cargo if p.active]
            
        if targets:
            closest_pkg = min(targets, key=lambda p: abs(rx - p.row) + abs(ry - p.col))
            px, py = closest_pkg.pos
            current_dist = abs(rx - px) + abs(ry - py)
            new_dist = abs(x - px) + abs(y - py)
    
            # 若新位置比舊位置更靠近該目標，給予獎勵
            if new_dist < current_dist:
                reward += 0.4
            
        # 判斷遊戲是否結束 (Done)
        all_good_collected = True
        for pkg in self.limit_cargo:
            if pkg.active:
                all_good_collected = False
                break
                    
        for pkg in self.good_cargo:
            if pkg.active:
                all_good_collected = False
                break
        
        done = False
        if all_good_collected:
            done = True
            reward += 20 # 清空全場獎勵
            
        if self.steps >= self.cfg.MAX_STEPS:
            done = True  # 超過最大步數
            
        self.score += reward
        return reward, done

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.clock = pygame.time.Clock()
        self.cell_width = 64
        self.cell_height = 64
        self.window_size = (self.cell_width * self.map_size, self.cell_height * self.map_size)
        self.window_surface = pygame.display.set_mode(self.window_size) 
        
        # 載入圖片 (需確保 sprites 資料夾存在)
        try:
            base_path = path.dirname(__file__)
            sprite_dir = path.join(base_path, "sprites")
            self.sprites = {
                'bot': pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "bot.png")), (64,64)),
                'floor': pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "floor.png")), (64,64)),
                'limit': pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "limit.png")), (64,64)),
                'good': pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "good.png")), (64,64)),
                'bad': pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "bad.png")), (64,64))
            }
        except Exception as e:
            print(f"Sprite loading failed: {e}")
            self.render_on = False

    def render(self):
        if not self.render_on or self.window_surface is None:
            return

        # 處理 Pygame 事件 (避免視窗卡死)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pix_size = 64
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        # 繪製地圖與物件
        for r in range(self.map_size):
            for c in range(self.map_size):
                canvas.blit(self.sprites['floor'], (c*pix_size, r*pix_size))
        
        for i in range(self.cfg.CARGO_TYPES):
            for target in self.all_cargos[i]:
                if target.active:
                    canvas.blit(self.sprites[target.name], (target.col * pix_size, target.row * pix_size))
              
        r_row, r_col = self.robot_pos
        canvas.blit(self.sprites['bot'], (r_col * pix_size, r_row * pix_size))

        self.window_surface.blit(canvas, canvas.get_rect())
        pygame.display.update()
        # self.clock.tick(30) # 如果需要限制 FPS

# 單元測試：如果不透過 Gym，直接跑這個檔案也能玩
if __name__ == "__main__":
    game = CargoGame(render_on=True)
    game.render()
    while True:
        # 隨機亂跑測試
        action = random.choice(list(RobotAction))
        _, done = game.perform_action(action)
        game.render()
        if done:
            game.reset()
