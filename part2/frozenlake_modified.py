import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

total_win = 0

def run(episodes, is_training=True, render=False):
    global total_win

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        try:
            f = open('frozen_lake8x8.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        except FileNotFoundError:
            print("❌ 找不到模型檔案！")
            return
    
    # 1. 學習率：動態衰減 
    start_lr = 0.7
    min_lr = 0.01
    
    # 2. 獎懲機制
    hole_penalty = -0.2
    step_penalty = -0.001
    dest_reward = 1.5

    # 3. Gamma
    discount_factor_g = 0.97

    # 4. 探索率：指數衰減
    epsilon = 1.0
    min_exploration_rate = 0.01
    epsilon_decay_rate = 0.9995
    
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        # === 動態學習率公式 ===
        # 隨著訓練次數增加，LR 逐漸變小，減少後期的震盪
        current_lr = max(min_lr, start_lr * (1 - i / (episodes * 0.9)))

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                max_value = np.max(q[state, :])
                best_actions = np.where(q[state, :] == max_value)[0]
                action = rng.choice(best_actions)

            new_state, reward, terminated, truncated, _ = env.step(action)

            # 獎勵機制
            if terminated and reward == 1:
                custom_reward = dest_reward    # 終點
            elif terminated and reward == 0:
                custom_reward = hole_penalty   # 掉洞 
            else:
                custom_reward = step_penalty   # 走路

            # 增加達成目標的學習率
            if custom_reward > 0:
                current_lr = 0.9
            if is_training:
                # 處理 Truncated (超時) : 保留未來價值
                if truncated:
                    target = custom_reward + discount_factor_g * np.max(q[new_state,:])
                elif terminated:
                    target = custom_reward
                else:
                    target = custom_reward + discount_factor_g * np.max(q[new_state,:])

                # 更新 Q Table
                q[state,action] = q[state,action] + current_lr * (target - q[state,action])

            state = new_state
            
            if reward == 1:
                rewards_per_episode[i] = 1

        # 衰減探索率
        epsilon = max(min_exploration_rate, epsilon * epsilon_decay_rate)

    env.close()

    # 繪圖
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.title(f'Target 70%: LR Decay, Hole -0.1')
    plt.savefig('frozen_lake8x8.png')
    
    if is_training == False:
        win = np.sum(rewards_per_episode) / episodes * 100
        print(f"✅ Success Rate: {win:.2f}% ({int(np.sum(rewards_per_episode))} / {episodes} episodes)")
        total_win += win

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    print("🔥 開始修正後訓練 (LR Decay, Hole -0.1)...")
    run(15000, is_training=True, render=False)

    print("\ntesting section (1000 times each round):")
    for i in range(0, 10):
        run(1000, is_training=False, render=False)
    
    print(f"\nFinal average success rate: {total_win / 10:.2f}%")
