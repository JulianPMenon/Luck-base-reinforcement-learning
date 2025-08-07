import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environments.minigrid_wrapper import MiniGridWrapper
from src.models.rl_agent import RLAgent
from src.utils.metrics import MetricsTracker

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import minigrid


def fill_b(agent:RLAgent, env):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    #print(f"Episode {episode} - Initial Observation stats: mean={observation.mean():.4f}, std={observation.std():.4f}, shape={observation.shape}")
    while len(agent.memory) < agent.max_memory:
        observation = env.reset()
        state = observation / 255.0  # Normalize the state
        done = False

        steps = 0
        mem = []
        while done == False and steps < 100:
            action = agent.act(state.to(device))
            next_observation, reward, done, truncated, _ = env.step(action)
            next_state = next_observation / 255.0  # Normalize the next state
            mem.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1
        if steps < 100:
            for state, action, reward, next_state, done in mem:
                agent.remember(state, action, reward, next_state, done)

def create_agent(rng):
    seed = 666666 #demo seed for testing
    env = MiniGridWrapper('MiniGrid-Empty-5x5-v0', seed=seed, cnn=True)
    state_size = env.observation_space.shape[0]
    agent = RLAgent(state_size=state_size, action_size=env.action_space.n, batch_size=128, epsilon = rng[0]*0.5+0.5, epsilon_decay=rng[1], epsilon_min=rng[2]*0.1, gamma=rng[3], learning_rate=rng[4]*0.2, intrinsic_weight=rng[5], hidden_size = 1 + int(rng[6]*256))
    return {'agent':agent, 'config':rng, 'reward':0.0}

def _create_agent(rng):
    seed = 666666 #demo seed for testing
    env = MiniGridWrapper('MiniGrid-Empty-5x5-v0', seed=seed, cnn=True)
    state_size = env.observation_space.shape[0]
    agent = RLAgent(state_size=state_size, action_size=env.action_space.n, batch_size=128, epsilon = rng[0], epsilon_decay=rng[1], epsilon_min=rng[2], gamma=rng[3], learning_rate=rng[4], intrinsic_weight=rng[5], hidden_size = int(rng[6]))
    return {'agent':agent, 'config':rng, 'reward':0.0}

def plot_heatMap(pop, env, actionmap, batch_size=32, plot = True):
    symbolmap = {0: "L", 1: "R", 2: "W", 3: "P", 4: "D", 5: "T", 6: "E"}
    agent = pop['agent']
    grid_size_width = env.env.unwrapped.width  # assumes square grid
    grid_size_height = env.env.unwrapped.height  # assumes square grid
    heatmap = np.zeros((grid_size_width*2, grid_size_height*2))
    actionmap_plot = heatmap.copy()
    # Save current agent position and direction
    orig_pos = tuple(env.env.unwrapped.agent_pos)
    orig_dir = env.env.unwrapped.agent_dir
    # x = height
    for x in range(grid_size_height):
        # y = width
        for y in range(grid_size_width):
            #print(state)
            #if state.shape[0] != agent.expected_state_size:
            #    print(f"State shape mismatch at ({x},{y}): got {state.shape[0]}, expected {agent.expected_state_size}")
            #    continue
            with torch.no_grad():
                enumerate({(0,0),(0,1),(1,0),(1,1)})

                #####
                #(0,0)^  (1,0)->
                #(0,1)<- (1,1)|
                for n, (x_add, y_add) in enumerate([(0,1),(1,1),(1,0),(0,0)]):
                    observation, _ = env.env.reset()
                    env.env.unwrapped.agent_pos = (x, y)
                    env.env.unwrapped.agent_dir = n
                    observation = env.env.observation(observation)
                    state = env.preprocess_observation(observation['image'])# / 255.0
                    q_values = agent.q_forward(state/batch_size)
                    max_q = round(q_values[0].max().item(), 2)
                    heatmap[x * 2 + x_add, y * 2 + y_add] = max_q  # y is row, x is col
                    actionmap_plot[x * 2 + x_add, y * 2 + y_add] = actionmap[int(q_values[0].argmax())]


    #print(heatmap)
    #print(actionmap_plot)
    #print(np.array([[symbolmap[action] for action in action_layer]for action_layer in actionmap_plot]))
    if plot:
        observation, _ = env.env.reset()
        env.env.unwrapped.agent_dir = 0
        observation = env.env.observation(observation)
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(observation['image'])
        sns.heatmap(ax=ax2, data = heatmap, square=True, annot=np.array([[symbolmap[action] for action in action_layer]for action_layer in actionmap_plot]), fmt='')


        for i in range(grid_size_height + 1):
            b, t = plt.ylim()
            ax2.vlines(x = (i) * 2, ymin = b, ymax = t, lw = 1, colors = 'black')

        
        for j in range(grid_size_width + 1):
            b, t = plt.xlim()
            ax2.hlines(y = (j) * 2, xmin = b, xmax = t, lw = 1, colors = 'black')
#        fig.colorbar(ax2)
        plt.show()

    # # Restore original position and direction
    # env.env.unwrapped.agent_pos = np.array(orig_pos)
    # env.env.unwrapped.agent_dir = orig_dir
    # plt.figure(figsize=(6, 5))
    # plt.imshow(heatmap, origin='lower', cmap='viridis')
    # plt.colorbar(label='Max Q-value')
    # plt.title("title")
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.tight_layout()
    # plt.show()

def train(pop, budget, plot, position, max):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    rng = pop['config']
    agent = pop['agent']
    print(f"New Config Budget: {budget} | {position+1} of {max}\nepsilon: {(rng[0]*0.5+0.5):.4} | epsilon_decay: {(rng[1]):.4} | epsilon_min: {(rng[2]*0.1):.4} | gamma: {rng[3]:.4} | learning_rate: {(rng[4]*0.2):.4} | intrinsic_weight: {rng[5]:.4} | hidden_size: {1+int(rng[6]*256)}")
    seed = 666666 #demo seed for testing
    env = MiniGridWrapper('MiniGrid-Empty-5x5-v0', seed=seed, cnn=True)
    agent.to(device)
    metrics = MetricsTracker()
    fill_b(agent, env)
    for episode in range(budget):
        
        observation = env.reset()
        #print(f"Episode {episode} - Initial Observation stats: mean={observation.mean():.4f}, std={observation.std():.4f}, shape={observation.shape}")
        state = observation / 255.0  # Normalize the state
        done = False
        steps = 0
        total_reward = 0
        while not done and steps < 100:
            action = agent.act(state)
            next_observation, reward, done, truncated, _ = env.step(action)
            next_state = next_observation / 255.0  # Normalize the next state
            agent.remember(state, action, reward, next_state, done)
            #print(reward)
            loss = 0
            if len(agent.memory) > agent.batch_size:    
                loss = agent.train()
                #print(loss)
                if loss > 0:
                    metrics.update_loss(loss_type='rl', loss=loss)
                    
            total_reward += reward
            state = next_state
            steps += 1

            if reward != 0:
                print(f"  Step {steps} | Reward: {reward:.2f} | loss {loss} | Epoch: {episode}")
        metrics.update_episode(total_reward, steps, [])

        if episode % 20 == 0:
            avg_reward = metrics.get_average_return(20)
            #print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Total Rewards: {total_reward:.2f}, done: {done}, Steps: {steps}")
            #if round(avg_reward,2) == 0.00:
                #break
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    avg_reward = metrics.get_average_return(20)
    print(f"RL Agent test completed successfully! {avg_reward}\n")

    if plot == True:
        plot_heatMap(pop,env)
    return {'agent':agent,'config':rng, 'reward':avg_reward}


def getreward(d):
    return d["reward"]


if __name__ == '__main__':    
    exponet = 6
    budget = 2 ** (exponet - 1)
    0,1,2,3,4 ,5 ,6 ,7
    1,2,4,8,16,32,64,128
    population = [create_agent(torch.rand(7)) for _ in range(budget)]
    best = []
    plot = True
    seed = 666666
    env = MiniGridWrapper('MiniGrid-Empty-5x5-v0', seed=seed, cnn=True)
    #epsilon: 0.827 | epsilon_decay: 0.1498 | epsilon_min: 0.07242 | gamma: 0.5619 | learning_rate: 0.09753 | intrinsic_weight: 0.8114 | hidden_size: 9
    pop = _create_agent([0.827, 0.1498, 0.07242, 0.5619, 0.09753, 0.8114, 9.])
    plot_heatMap(pop,env,{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8})
    # #train(pop,128,True,1,1)
    # #for n in range(exponet):
    #     if n == exponet-1:
    #         best = population[0]
    #         plot = True
    #     print("Budget", 2**n)
    #     population = [train(pop, 2**n, plot, position, int(2**exponet/2**(n-1))) for position, pop in enumerate(population)]
    #     population.sort(key=getreward, reverse=True)
    #     population = [pop if pop['reward'] > 0.00 else create_agent(torch.rand(7)) for pop in population[0:int(budget/2**(n+1))]]
    # print(population[0])
    #config = [0.4623, 0.5264, 0.6604, 0.6262, 0.07074, 0.9464]  
    #train(best, 2**exponet)
    from torch import nn
    m = nn.Softmax(dim=1)
    input = torch.randn(2, 3)
    print(input)
    output = m(input)
    print(output)
    

