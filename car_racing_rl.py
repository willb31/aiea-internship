import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os

# Create log directory
log_dir = "./car_racing_logs/"
os.makedirs(log_dir, exist_ok=True)

# Create the environment
env = gym.make("CarRacing-v3", render_mode="rgb_array")
env = Monitor(env, log_dir)

# Create the PPO model with TensorBoard logging
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./car_racing_tensorboard/"
)

# Train the model
print("Starting training...")
model.learn(total_timesteps=100000, tb_log_name="ppo_car_racing")

# Save the model
model.save("ppo_car_racing")
print("Training complete! Model saved.")

env.close()