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
from unit_tests import random_search
import torch.optim as optim
import torch.nn.functional as F

def load_config(config_path):
    """Load configuration from a YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def generate_memorybank(config, contrastiv_rl_agent:Contrastiv_RL_agent):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running experiment: {config['name']}")
    
    
    # 1. Initialize environment
    env = MiniGridWrapper(config['env_name'], seed= 42)
    
    # 2. Collect data for contrastive learning
    print("                                           |   mmm   |\n"
          "Collecting data for contrastive learning...|  (@_@)  |\n"
          "                                           | <( : )> |\n"
          "                                           |   / \   |\n"
          )
    
    data_collector = DataCollector(max_episodes=config['data_collection_episodes'], max_steps_per_episode=config['max_steps_per_episode'])
    observations = data_collector.collect_data(env)
    queries, keys = data_collector.create_contrastive_pairs(observations, mode="NOISE")
    queries = queries.to(device)
    keys = keys.to(device)
    
    # 3. Initialize contrastive model
    state_dim = config['latent_dim']  # Use latent representation
    action_dim = len(config['actionmap'])
    contrastiv_rl_agent.to(device)
    # contrastive_model = ContrastiveLearningAgent(
    #     input_channels=3,  # Assuming RGB images
    #     latent_dim=config['latent_dim']
    # )
    
    # 4. Train contrastive model
    print("                                           |   mmm   |\n"
          "Training contrastive model...              |  (O.O)  |\n"
          "                                           | <( : )> |\n"
          "                                           |   / \   |\n"
          )
    #contrastive_trainer = ContrastiveTrainer(contrastive_model)
    contrastive_losses = contrastiv_rl_agent.train_contrastive_model(
        queries, keys, 
        epochs=config['contrastive_epochs']
    )
    
    # 5. Initialize RL agent
    # state_dim = config['latent_dim']  # Use latent representation
    # action_dim = env.action_space.n
    # agent = RLAgent(state_dim, action_dim)
    
    # 6. Build memory bank for entropy estimation
    print("                                           |  Bank  |\n"
          "Building memory bank...                    |        |\n"
          "                                           |   __   |\n"
          "                                           |   ||   |\n"
          )
    memory_bank = []
    with torch.no_grad():
        for i in range(0, len(observations)):  # Sample every 100th observation
            obs_batch = torch.stack(observations[i:i+10]).to(device)
            encodings = contrastiv_rl_agent.query_encoder(obs_batch)
            memory_bank.extend(encodings)
    memory_bank = torch.stack(memory_bank)
    return memory_bank

def run_experiment(config: dict, contrastiv_rl_agent:Contrastiv_RL_agent, metrics,max_episodes:int=0, memory_bank=[]):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running experiment: {config['name']}, {config['epsilon']}, {config['epsilon_decay']}, {config['gamma']}")
    env = MiniGridWrapper(config['env_name'], seed= 42)
    contrastiv_rl_agent.to(device)
    if memory_bank == []:
        memory_bank = generate_memorybank(config=config,contrastiv_rl_agent=contrastiv_rl_agent)
    
    random_search.plot_heatMap(pop = {'agent': contrastiv_rl_agent}, env = env, actionmap=config['actionmap'], plot=False)
    
    best_agent = copy.deepcopy(contrastiv_rl_agent)
    best_avg_reward = 0
    best_epoch = 0
    best_epsilon = 0


    # 7. Training
    print("Training RL agent with intrinsic rewards...")
    avg_reward = 0
    optimizer = optim.Adam(contrastiv_rl_agent.rl_agent.q_network.parameters(), lr=0.00001)
    max_episode=0
    if max_episodes == 0:
        max_episode = config['rl_episodes']
    else:
        max_episode = max_episodes
    # --- Add TD-InfoNCE optimizer outside training loop ---
    td_optimizer = optim.Adam(contrastiv_rl_agent.contrastive_model.parameters(), lr=0.0001)
    td_infonce_interval = 10
    for episode in range(max_episode+1):
        obs = env.reset()
        total_reward = 0
        steps = 0
        visited_states = []
        rl_transitions = []           # For RL/HER (detached)
        contrastive_transitions = []  # For TD-InfoNCE (no detach)
        done = False
        lastloss = -1
        while not done and steps < config['max_steps_per_episode']:
            # TD-InfoNCE: encode with grad enabled
            state_encoding = contrastiv_rl_agent.query_encoder(obs.unsqueeze(0)).squeeze().to(device)
            visited_states.append(state_encoding)
            action = contrastiv_rl_agent.act(state_encoding)

            next_obs, reward, terminated, truncated, _ = env.step(config['actionmap'][action])
            done = terminated or truncated

            # Intrinsic reward and next_state_encoding for RL/HER (no grad needed)
            with torch.no_grad():
                intrinsic_reward = contrastiv_rl_agent.compute_state_entropy(
                    next_obs.unsqueeze(0), memory_bank
                ).item()
                next_state_encoding_nograd = contrastiv_rl_agent.query_encoder(next_obs.unsqueeze(0)).squeeze().to(device)

            # TD-InfoNCE: encode next state with grad enabled
            next_state_encoding = contrastiv_rl_agent.query_encoder(next_obs.unsqueeze(0)).squeeze().to(device)

            # Store transitions
            rl_transitions.append((state_encoding.detach(), action, reward, next_state_encoding_nograd.detach(), done))
            contrastive_transitions.append((state_encoding, next_state_encoding))

            contrastiv_rl_agent.remember(
                state_encoding.detach(), action, reward,
                next_state_encoding_nograd.detach(), done, intrinsic_reward
            )

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

        # --- Hindsight Experience Replay (HER) ---
        K = 4
        for t, (state, action, reward, next_state, done) in enumerate(rl_transitions):
            future_idxs = torch.randint(t, len(rl_transitions), (K,))
            for idx in future_idxs:
                new_goal = rl_transitions[idx][3]
                her_reward = 1.0 if torch.norm(next_state - new_goal) < 1e-3 else 0.0
                contrastiv_rl_agent.remember(state, action, her_reward, next_state, done, intrinsic_reward=0)

        # --- TD-InfoNCE update ---
        if episode % td_infonce_interval == 0 and len(contrastive_transitions) > config['batch_size']:
            batch_idxs = torch.randint(0, len(contrastive_transitions), (config['batch_size'],))
            batch_states = torch.stack([contrastive_transitions[i][0] for i in batch_idxs]).to(device)
            batch_next_states = torch.stack([contrastive_transitions[i][1] for i in batch_idxs]).to(device)
            query = F.normalize(batch_states, dim=1)
            key = F.normalize(batch_next_states, dim=1)
            logits = torch.mm(query, key.t()) / 0.1
            labels = torch.arange(query.size(0), device=query.device)
            td_loss = F.cross_entropy(logits, labels)
            td_optimizer.zero_grad()
            td_loss.backward()
            td_optimizer.step()
            if hasattr(metrics, 'update_loss'):
                metrics.update_loss(loss_type='td_infonce', loss=td_loss.item())

        #if episode % 100 == 0:
        contrastiv_rl_agent.update_target_network()

        if episode % 10 == 0:
            contrastiv_rl_agent.epsilon_decay()

        metrics.update_episode(total_reward, steps, visited_states)

        if episode % 10 == 0:
            avg_reward = metrics.get_average_return()
            exploration_efficiency = metrics.get_exploration_efficiency()
            print(f"Episode {episode}: Total Reward: {total_reward}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Exploration Efficiency: {exploration_efficiency:.2f},"
                  f"Last loss = {lastloss}")
            random_search.plot_heatMap(pop = {'agent': contrastiv_rl_agent}, env = env, actionmap=config['actionmap'], plot=False)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_agent = copy.deepcopy(contrastiv_rl_agent)
                best_epoch = episode
                best_epsilon = best_agent.rl_agent.epsilon
            
    # 8. Evaluation
    result_dir = f"results/{config['name']}"
    os.makedirs(result_dir, exist_ok=True)
    #metrics.plot_metrics(save_path=f"{result_dir}/metrics.png")
    #torch.save(contrastiv_rl_agent.contrastive_model.state_dict(), f"{result_dir}/contrastive_model.pth")
    #torch.save(contrastiv_rl_agent.rl_agent.state_dict(), f"{result_dir}/rl_agent.pth")
    print(f"Experiment {config['name']} completed. Results saved to {result_dir}")
    #random_search.plot_heatMap(pop = {'agent': contrastiv_rl_agent}, env = env, actionmap=config['actionmap'])
    return {'agent':best_agent, 'config':config, 'avg_reward':best_avg_reward, 'memory_bank':memory_bank, 'metrics':metrics, 'epoch':best_epoch, 'epsilon':best_epsilon}

if __name__ == "__main__":
    config = load_config('rl_project/experiments/configs/moderate_task.yaml')
    result_dir = f"results/{config['name']}/contrastiv"
    with open(result_dir+"/results.txt", 'a') as f:
        for seed in range(1):
            print(seed)
            torch.manual_seed(seed)
            state_dim = config['latent_dim']  # Use latent representation
            action_dim = len(config['actionmap']) 
            contrastiv_rl_agent = Contrastiv_RL_agent(state_dim, action_dim, input_channels=3,latent_dim=config['latent_dim'], epsilon= config['epsilon'], epsilon_decay=config['epsilon_decay'], gamma = config['gamma'])
            metrics = MetricsTracker()
            population = run_experiment(config,contrastiv_rl_agent=contrastiv_rl_agent, metrics=metrics)
            random_search.plot_heatMap(pop = {'agent': population['agent']}, env = MiniGridWrapper(population['config']['env_name'], seed= 42), actionmap=population['config']['actionmap'],save_path=f"{result_dir}/heat_map_{seed}_.png")
            population['metrics'].plot_metrics(save_path=f"{result_dir}/metrics_{seed}_.png")
            print(f"seed: {seed} | epoch: {population['epoch']} | epsilon: {population['epsilon']} | avg_reward: {population['avg_reward']} | len(memory_bank): {len(population['memory_bank'])}")
            f.write(f"seed: {seed} | epoch: {population['epoch']} | epsilon: {population['epsilon']} | avg_reward: {population['avg_reward']} | len(memory_bank): {len(population['memory_bank'])}\n")
            torch.save(contrastiv_rl_agent.contrastive_model.state_dict(), f"{result_dir}/contrastive_model_{seed}.pth")
            torch.save(contrastiv_rl_agent.rl_agent.state_dict(), f"{result_dir}/rl_agent_{seed}.pth")

