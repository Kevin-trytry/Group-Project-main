import gymnasium as gym
import numpy as np
import random
import pygame
import os
from os import path
from config import Config
from cargo import GoodCargo, BadCargo, LimitedCargo

class Environment:
    def __init__(self, Config, render_mode=None):
        self.cfg = Config
        self.render_mode = render_mode
        self.map_size = self.cfg.MAP_SIZE
        self.state_dim = (4, self.cfg.MAP_SIZE, self.cfg.MAP_SIZE) 
        self.action_space = 4 
        
        if self.render_mode == 'human':
            self._init_pygame()       
        
        self.reset()
    
    def reset(self):
        self.steps = 0
        
        positions = random.sample(range(self.map_size * self.map_size), 7) 
        coordinates = [self._to_coord(pos) for pos in positions]
        
        self.robot_pos = tuple(coordinates[0])
        self.limit_cargo = LimitedCargo(*coordinates[1])
        self.good_cargo = [GoodCargo(*coordinates[2]), GoodCargo(*coordinates[3]), GoodCargo(*coordinates[4])]
        self.bad_cargo = [BadCargo(*coordinates[5]), BadCargo(*coordinates[6])]
        
        return self._get_observation()
    
    def _to_coord(self, idx):
        return (idx // self.map_size, idx % self.map_size)
    
    def _get_observation(self):
        obs = np.zeros((4, self.map_size, self.map_size), dtype=np.float32)
        
        rx, ry = self.robot_pos
        obs[0, rx, ry] = 1.0
        
        if self.limit_cargo.active:
            lx, ly = self.limit_cargo.pos
            obs[1, lx, ly] = self.limit_cargo.remain_lifetime / self.limit_cargo.lifetime
            #obs[1, lx, ly] = 1.0
            
        for item in self.good_cargo:
            if item.active:
                obs[2, item.row, item.col] = 1.0
            
        for item in self.bad_cargo:
            if item.active:
                obs[3, item.row, item.col] = 1.0
            
        return obs
    
    def step(self, action):
        self.steps += 1
        rx, ry = self.robot_pos
        x, y = rx, ry
        
        # 0:上, 1:下, 2:左, 3:右
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1
        
        reward = self.cfg.STEP_COST
        
        # 邊界檢查 (撞牆)
        if x < 0 or x >= self.map_size or y < 0 or y >= self.map_size:
            reward += -1 
        else:
            self.robot_pos = (x, y)

        # 檢查高分包裹
        self.limit_cargo.update()
        if self.limit_cargo.active:
            if self.robot_pos == self.limit_cargo.pos:
                reward += self.limit_cargo.get_reward()
        
        # 檢查一般包裹
        for pkg in self.good_cargo:
            if pkg.active and self.robot_pos == pkg.pos:
                reward += pkg.get_reward()
            
        # 檢查負分包裹
        for pkg in self.bad_cargo:
            if pkg.active and self.robot_pos == pkg.pos:
                reward += pkg.get_reward()
            
        all_good_collected = True
        if self.limit_cargo.active:
            all_good_collected = False
            
        for pkg in self.good_cargo:
            if pkg.active:
                all_good_collected = False
                break
        
        done = False
        if all_good_collected:
            done = True
            reward += 20 # 給予清空全場的額外獎勵
            
        if self.steps >= self.cfg.MAX_STEPS:
            done = True
            
        return self._get_observation(), reward, done
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            
    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.clock = pygame.time.Clock()
        self.action_font = pygame.font.SysFont("Calibre",30)
        self.cell_width = 64
        self.cell_height = 64
        self.window_size = (self.cell_width * self.map_size, self.cell_height * self.map_size)
        self.window_surface = pygame.display.set_mode(self.window_size) 
        
        try:
            base_path = path.dirname(__file__)
            sprite_dir = path.join(base_path, "sprites")
            self.robot_img = pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "bot.png")), (64,64))
            self.floor_img = pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "floor.png")), (64,64))
            self.limit_img = pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "limit.png")), (64,64)) 
            self.good_img = pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "good.png")), (64,64)) 
            self.bad_img = pygame.transform.scale(pygame.image.load(path.join(sprite_dir, "bad.png")), (64,64)) 
        except:
            print("Sprite loading failed, please check path.")

    def _render_frame(self):
        if not hasattr(self, 'window_surface'): return
        pix_size = 64
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        for r in range(self.map_size):
            for c in range(self.map_size):
                canvas.blit(self.floor_img, (c*pix_size, r*pix_size))
              
        if self.limit_cargo.active:
            canvas.blit(self.limit_img, (self.limit_cargo.col * pix_size, self.limit_cargo.row * pix_size))
        
        for target in self.good_cargo:
            if target.active:
                canvas.blit(self.good_img, (target.col * pix_size, target.row * pix_size))
        
        for target in self.bad_cargo:
            if target.active:
                canvas.blit(self.bad_img, (target.col * pix_size, target.row * pix_size))
                
        r_row, r_col = self.robot_pos
        canvas.blit(self.robot_img, (r_col * pix_size, r_row * pix_size))

        self.window_surface.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()