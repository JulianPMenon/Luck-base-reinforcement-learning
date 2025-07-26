#!/bin/bash
# setup_environment.sh

echo "Setting up RL Project Environment..."

# Uninstall old packages if they exist
echo "Removing old gym-minigrid if installed..."
pip uninstall gym-minigrid -y 2>/dev/null || true

# Install required packages
echo "Installing required packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium>=0.26.0
pip install minigrid>=2.0.0
pip install numpy matplotlib pyyaml tqdm scikit-learn

# Verify installation
echo "Verifying installation..."
python -c "
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
print('✅ All packages installed successfully!')

# Test basic functionality
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode='rgb_array')
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)
obs, info = env.reset()
print(f'✅ Environment test successful! Obs shape: {obs.shape}')
env.close()
"

echo "✅ Setup complete! You can now run the project."