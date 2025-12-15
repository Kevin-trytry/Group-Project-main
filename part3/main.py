import torch
import time
from environment import CargoEnv as Environment
from ddqn_agent import DQNAgent
from config import Config

def run(is_training=True, render=False):
    """
    統一的執行入口
    :param is_training: True 代表訓練模式, False 代表測試/展示模式
    :param render: 是否要顯示畫面 (訓練時建議 False 以加快速度)
    """
    config = Config()
    
    # 根據參數決定是否要開啟畫面
    render_mode = 'human' if render else None
    
    env = Environment(render_mode=render_mode)
    
    agent = DQNAgent(config)
    
    model_path = "cargo_dqn_model.pth"

    if is_training:
        print(f"Start Training on device: {agent.device}")
        print("-" * 30)
        
        total_steps = 0
        
        for i_episode in range(config.EPISODES):
            state, _ = env.reset() 
            total_reward = 0
            
            while True:
                # 1. 選擇動作
                action = agent.select_action(state)
                
                # (obs, reward, terminated, truncated, info)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # 合併 terminated 和 truncated 為 done
                done = done or truncated
                
                # 2. 存入記憶體
                agent.memory.push(state, action, reward, next_state, done)
                
                # 3. 更新網路參數
                total_steps += 1
                if total_steps % 4 == 0:
                    agent.update()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # 每個 Episode 結束後：
            agent.update_epsilon() # 降低探索率
            
            # 更新 Target Network
            if i_episode % config.TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # 顯示進度
            if i_episode % 100 == 0:
                print(f"Episode {i_episode}\t Score: {total_reward:.2f}\t Epsilon: {agent.epsilon:.2f}")

        print("Training Finished!")
        torch.save(agent.policy_net.state_dict(), model_path)
        print(f"Model saved as '{model_path}'")

    else:
        try:
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
            agent.policy_net.eval()
            print("成功載入模型！")
        except FileNotFoundError:
            print(f"Cannot find the model {model_path}, please set is_training=True to train the model first.")
            return

        agent.epsilon = 0.05 
        
        test_episodes = 5
        for i_episode in range(test_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                state = next_state
                total_reward += reward
                
                if render:
                    # 可以稍微調慢一點才看得到動作
                    time.sleep(0.05) 
                    env.render()

            print(f"Demo Episode {i_episode+1}/{test_episodes} Score: {total_reward:.2f}")

if __name__ == "__main__":
    # 1. 訓練模式
    run(is_training=True, render=False)

    # 2. 測試模式 (要看畫面請把這裡取消註解，並把上面註解掉)
    run(is_training=False, render=True)