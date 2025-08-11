import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
import copy
from src.environments.minigrid_wrapper import MiniGridWrapper
from src.utils.data_collection import DataCollector
from src.utils.metrics import MetricsTracker
from src.models.contrastiv_rl_agent import Contrastiv_RL_agent
from unit_tests import hyperband
import torch.optim as optim

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def generate_memorybank(config, contrastiv_rl_agent:Contrastiv_RL_agent):
    """
    collects and augments data. trains encoder and generates the memorybank 
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    #create environment
    print(f"Running experiment: {config['name']}")
    env = MiniGridWrapper(config['env_name'], seed=42)
    
    #collect data for training
    print("Collecting data for contrastive learning...")
    data_collector = DataCollector(max_episodes=config['data_collection_episodes'], max_steps_per_episode=config['max_steps_per_episode'])
    observations = data_collector.collect_data(env)
    queries, keys = data_collector.create_contrastive_pairs(observations, mode="NOISE")
    queries = queries.to(device)
    keys = keys.to(device)
    state_dim = config['latent_dim']
    action_dim = len(config['actionmap'])
    contrastiv_rl_agent.to(device)
    print("Training contrastive model...")
    contrastive_losses = contrastiv_rl_agent.train_contrastive_model(
        queries, keys, 
        epochs=config['contrastive_epochs']
    )
    
    #build memory bank
    print("Building memory bank...")
    memory_bank = []
    with torch.no_grad():
        for i in range(0, len(observations)):
            obs_batch = torch.stack(observations[i:i+10]).to(device)
            encodings = contrastiv_rl_agent.query_encoder(obs_batch)
            memory_bank.extend(encodings)
    memory_bank = torch.stack(memory_bank)
    return memory_bank

def run_experiment_no_her(config: dict, contrastiv_rl_agent:Contrastiv_RL_agent, metrics,max_episodes:int=0, memory_bank=[]):
    """
    runs training for fixed agent and episodes
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    print(f"Running experiment (no HER): {config['name']}, {config['epsilon']}, {config['epsilon_decay']}, {config['gamma']}")
    env = MiniGridWrapper(config['env_name'], seed=42)
    contrastiv_rl_agent.to(device)
    # initialize memorybank
    if memory_bank == []:
        memory_bank = generate_memorybank(config=config,contrastiv_rl_agent=contrastiv_rl_agent)
    
    # remember best config
    best_agent = copy.deepcopy(contrastiv_rl_agent)
    best_avg_reward = 0
    best_epoch = 0
    best_epsilon = 0
    print("Training RL agent with intrinsic rewards (no HER)...")
    avg_reward = 0

    # start trainig
    optimizer = optim.Adam(contrastiv_rl_agent.rl_agent.q_network.parameters(), lr=0.00001)
    max_episode=0
    if max_episodes == 0:
        max_episode = config['rl_episodes']
    else:
        max_episode = max_episodes
    for episode in range(max_episode+1):
        obs = env.reset()
        total_reward = 0
        steps = 0
        visited_states = []
        done = False
        lastloss=-1
        # run one game
        while not done and steps < config['max_steps_per_episode']:
            with torch.no_grad():
                state_encoding = contrastiv_rl_agent.query_encoder(obs.unsqueeze(0))
                visited_states.append(state_encoding.squeeze())
            action = contrastiv_rl_agent.act(state_encoding.squeeze())
            next_obs, reward, terminated, truncated, _ = env.step(config['actionmap'][action])
            done = terminated or truncated
            with torch.no_grad():
                intrinsic_reward = contrastiv_rl_agent.compute_state_entropy(
                    next_obs.unsqueeze(0), memory_bank
                ).item()
            with torch.no_grad():
                next_state_encoding = contrastiv_rl_agent.query_encoder(next_obs.unsqueeze(0))

            # remember the current stateinteraction
            contrastiv_rl_agent.remember(
                state_encoding.squeeze(), action, reward, 
                next_state_encoding.squeeze(), done, intrinsic_reward
            )

            # train 100 iterations
            if len(contrastiv_rl_agent.memory()) > 100:
                loss = contrastiv_rl_agent.train(optimizer)
                lastloss = loss
                try:
                    float_loss = float(loss)
                    metrics.update_loss(loss_type='rl', loss=float_loss)
                except (TypeError, ValueError):
                    print(f"[Warning] RL loss is not a float: {loss} (type: {type(loss)}) - skipping metrics update.")
            total_reward += reward
            steps += 1
            obs = next_obs
            metrics.update_intrinsic_reward(intrinsic_reward=intrinsic_reward)
        contrastiv_rl_agent.update_target_network()
        if episode % 10 == 0:
            contrastiv_rl_agent.epsilon_decay()
        metrics.update_episode(total_reward, steps, visited_states)
        if episode % 50 == 0:
            avg_reward = metrics.get_average_return()
            exploration_efficiency = metrics.get_exploration_efficiency()
            print(f"Episode {episode}: Total Reward: {total_reward}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Exploration Efficiency: {exploration_efficiency:.2f},"
                  f"Last loss = {lastloss}")
            hyperband.plot_heatMap(pop = {'agent': contrastiv_rl_agent}, env = env, actionmap=config['actionmap'], plot=False)
            # save the best config
            if avg_reward > best_avg_reward and episode != 0:
                best_avg_reward = avg_reward
                best_agent = copy.deepcopy(contrastiv_rl_agent)
                best_epoch = episode
                best_epsilon = best_agent.rl_agent.epsilon
    result_dir = f"results/{config['name']}_no_her"
    os.makedirs(result_dir, exist_ok=True)
    print(f"Experiment {config['name']} (no HER) completed. Results saved to {result_dir}")
    return {'agent':best_agent, 'config':config, 'avg_reward':avg_reward, 'best_avg_reward':best_avg_reward, 'memory_bank':memory_bank, 'metrics':metrics, 'epoch':best_epoch, 'epsilon':best_epsilon}

if __name__ == "__main__":
    config = load_config('rl_project/experiments/configs/moderate_task_lava_gap.yaml')
    result_dir = f"results/{config['name']}/contrastiv"
    os.makedirs(result_dir, exist_ok=True)
    with open(result_dir+"/results.txt", 'a') as f:
        # coose seeds
        for seed in range(1):
            print(seed)
            # set seed globaly
            torch.manual_seed(seed)
            state_dim = config['latent_dim']
            action_dim = len(config['actionmap']) 
            contrastiv_rl_agent = Contrastiv_RL_agent(state_dim, action_dim, input_channels=3,latent_dim=config['latent_dim'], epsilon= config['epsilon'], epsilon_decay=config['epsilon_decay'], gamma = config['gamma'])
            metrics = MetricsTracker()

            population = run_experiment_no_her(config,contrastiv_rl_agent=contrastiv_rl_agent, metrics=metrics)
            # save best config parametes
            hyperband.plot_heatMap(pop = {'agent': population['agent']}, env = MiniGridWrapper(population['config']['env_name'], seed= 42), actionmap=population['config']['actionmap'],save_path=f"{result_dir}/heat_map_{seed}_.png")
            population['metrics'].plot_metrics(save_path=f"{result_dir}/metrics_{seed}_.png")
            print(f"seed: {seed} | epoch: {population['epoch']} | epsilon: {population['epsilon']} | avg_reward: {population['best_avg_reward']} | len(memory_bank): {len(population['memory_bank'])}")
            f.write(f"seed: {seed} | epoch: {population['epoch']} | epsilon: {population['epsilon']} | avg_reward: {population['best_avg_reward']} | len(memory_bank): {len(population['memory_bank'])}\n")
            torch.save(contrastiv_rl_agent.contrastive_model.state_dict(), f"{result_dir}/contrastive_model_{seed}.pth")
            torch.save(contrastiv_rl_agent.rl_agent.state_dict(), f"{result_dir}/rl_agent_{seed}.pth")
