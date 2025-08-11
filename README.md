
# Luck-base-reinforcement-learning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JulianPMenon/Luck-base-reinforcement-learning.git
   cd Luck-base-reinforcement-learning
   ```

2. (Recommended) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r rl_project/requirements.txt
   ```

4. Install MiniGrid:
   ```bash
   pip install gym-minigrid
   ```

## Usage

To run the main experiment:

```bash
python rl_project/experiments/run_experiment.py
```

You can modify the configuration file path in `run_experiment.py` to select different MiniGrid environments or experiment settings.

You can change the configuration of our experiments int the corresponding yaml files at:
`rl_project/experiments/configs`

You can find the T-Test in `rl_project/utils/t_test.py`
If you run it uses our experiment results to calculate t and p values.

```bash
python rl_project/utils/t_test.py
```

You can try our hyperband (Whis is a little wonky)

```bash
python rl_project/unit_tests/hyperband.py
```
(Its in unit_tests because it was a test for different parameters first and then became a full Hyperband)
## Results

You can find our results for the 10 seeds per approach in `results/easy_task` or `results/moderate_task` (which is now referred as hard task in the paper)


## Main Files Used in run_experiment.py

Below is a list of the main files used by `run_experiment.py` and their locations:

1. **Main Experiment Script**
   - `rl_project/experiments/run_experiment.py`

2. **Configuration Files**
   - `rl_project/experiments/configs/` (YAML files, e.g., `easy_task.yaml`, `moderate_task.yaml`)

3. **MiniGrid Environment Wrapper**
   - `rl_project/src/environments/minigrid_wrapper.py`

4. **Data Collection Utilities**
   - `rl_project/src/utils/data_collection.py`

5. **Metrics Tracker**
   - `rl_project/src/utils/metrics.py`

6. **RL Agent/Model**
   - `rl_project/src/models/rl_agent.py` (or `contrastive_model.py` depending on experiment)

7. **Encoder**
   - `rl_project/src/models/encoder.py`

8. **Training Utilities**
   - `rl_project/src/training/` (e.g., `contrastive_trainer.py`, `train_CNN.py`)

9. **Results Output**
   - `results/` (e.g., `results/easy_task/`, `results/moderate_task/`)

## Reference

This project uses [MiniGrid](https://github.com/Farama-Foundation/MiniGrid) for RL environments.

For more details on MiniGrid, see their official documentation: https://minigrid.farama.org/

