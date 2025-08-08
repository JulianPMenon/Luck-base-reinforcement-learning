import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from src.environments.minigrid_wrapper import MiniGridWrapper
from src.utils.data_collection import DataCollector
from src.utils.metrics import MetricsTracker
from src.models.contrastiv_rl_agent2 import Contrastiv_RL_agent2
from unit_tests import random_search
import torch.optim as optim

def load_config(config_path):
    """Load configuration from a YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def run_experiment(config: dict, seed:int):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch.manual_seed(seed)
    print(f"Running experiment: {config['name']}")
    
    # 1. Initialize environment
    env = MiniGridWrapper(config['env_name'], seed= 42)
    
    # 2. Collect data for contrastive learning
    print("                                           |   mmm  \n"
          "Collecting data for contrastive learning...|  (@_@) \n"
          "                                           | <( : )>\n"
          "                                           |   / \  \n"
          )
    
    data_collector = DataCollector(max_episodes=config['data_collection_episodes'])
    observations = data_collector.collect_data(env)
    queries, keys = data_collector.create_contrastive_pairs(observations, mode="NOISE")
    queries = queries.to(device)
    keys = keys.to(device)
    
    # 3. Initialize contrastive model
    state_dim = config['latent_dim']  # Use latent representation
    action_dim = len(config['actionmap'])
    contrastiv_rl_agent = Contrastiv_RL_agent2(state_dim, action_dim, input_channels=3,latent_dim=config['latent_dim'], epsilon_decay=1-20/config['rl_episodes'], gamma = 1-4/config['rl_episodes']/100)
    contrastiv_rl_agent.to(device)
    # contrastive_model = ContrastiveLearningAgent(
    #     input_channels=3,  # Assuming RGB images
    #     latent_dim=config['latent_dim']
    # )
    
    # 4. Train contrastive model
    print("                                           |   mmm  \n"
          "Training contrastive model...              |  (O.O) \n"
          "                                           | <( : )>\n"
          "                                           |   / \  \n"
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
        for i in range(0, len(observations), 100):  # Sample every 100th observation
            obs_batch = torch.stack(observations[i:i+10]).to(device)
            encodings = contrastiv_rl_agent.query_encoder(obs_batch)
            memory_bank.extend(encodings)
    memory_bank = torch.stack(memory_bank)
    
    random_search.plot_heatMap(pop = {'agent': contrastiv_rl_agent}, env = env, actionmap=config['actionmap'], plot=False)
    # 7. Training
    print("Training RL agent with intrinsic rewards...")
    metrics = MetricsTracker()
    optimizer = optim.Adam(contrastiv_rl_agent.rl_agent.q_network.parameters(), lr=0.00001)
    for episode in range(config['rl_episodes']):
        obs = env.reset()
        total_reward = 0
        steps = 0
        visited_states = []
        episode_transitions = []  # For HER
        done = False
        lastloss=-1
        while not done and steps < config['max_steps_per_episode']:
            with torch.no_grad():
                state_encoding = contrastiv_rl_agent.query_encoder(obs.unsqueeze(0))
                visited_states.append(state_encoding.squeeze())
            #print(f"state encoding: {state_encoding.squeeze().shape}")
            action = contrastiv_rl_agent.act(state_encoding.squeeze())

            next_obs, reward, terminated, truncated, _ = env.step(config['actionmap'][action])
            done = terminated or truncated
            # if reward != 0:
            #     print(f"  Step {steps} | Reward: {reward:.2f} | Epoch: {episode}")
            #with torch.no_grad():
            #    intrinsic_reward = contrastiv_rl_agent.compute_state_entropy(
            #        next_obs.unsqueeze(0), memory_bank
            #    ).item()

            with torch.no_grad():
                next_state_encoding = contrastiv_rl_agent.query_encoder(next_obs.unsqueeze(0))

            # Store transition for HER: (state, action, reward, next_state, done, achieved_goal)
            achieved_goal = next_state_encoding.squeeze().detach().clone()
            episode_transitions.append((obs.detach().clone(), action, reward, next_obs.detach().clone(), done, achieved_goal))

            # Standard experience replay
            intrinsic_reward = contrastiv_rl_agent.remember(
                obs, action, reward, 
                next_obs, done
            )

            if len(contrastiv_rl_agent.memory()) > 100:
                loss = contrastiv_rl_agent.train(optimizer)
                lastloss = loss
                # Ensure only float loss values are appended
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
        K = 4  # Number of HER samples per transition
        for t, (state, action, reward, next_state, done, _) in enumerate(episode_transitions):
            future_idxs = torch.randint(t, len(episode_transitions), (K,))
            for idx in future_idxs:
                new_goal = episode_transitions[idx][3]  # Use future next_state_encoding as new goal
                # HER reward: 1 if next_state matches new_goal (within tolerance), else 0
                # Here, use L2 distance threshold
                her_reward = 1.0 if torch.norm(next_state - new_goal) < 1e-3 else 0.0
                # Store HER transition (no intrinsic reward for HER transitions)
                contrastiv_rl_agent.remember(state, action, her_reward, next_state, done, intrinsic_reward=0)

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
            
            
    # 8. Evaluation
    result_dir = f"results/{config['name']}"
    os.makedirs(result_dir, exist_ok=True)
    metrics.plot_metrics(save_path=f"{result_dir}/metrics.png")
    torch.save(contrastiv_rl_agent.contrastive_model.state_dict(), f"{result_dir}/contrastive_model.pth")
    torch.save(contrastiv_rl_agent.rl_agent.state_dict(), f"{result_dir}/rl_agent.pth")
    print(f"Experiment {config['name']} completed. Results saved to {result_dir}")
    random_search.plot_heatMap(pop = {'agent': contrastiv_rl_agent}, env = env, actionmap=config['actionmap'])
    return metrics

if __name__ == "__main__":
    run_experiment(load_config('rl_project/experiments/configs/easy_task.yaml'), 0)
