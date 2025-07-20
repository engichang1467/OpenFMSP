import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from env import CarTagEnv


def run_episode(env, num_steps=100):
    obs, _ = env.reset()

    for _ in range(num_steps):
        pursuer_action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
        evader_action = np.array([np.random.uniform(-np.pi, np.pi)], dtype=np.float32)
        action = {
            "pursuer_action": pursuer_action,
            "evader_action": evader_action
        }

        obs, reward, done, truncated, info = env.step(action)
        env.render()

        if done or truncated:
            break


if __name__ == "__main__":
    env = CarTagEnv(render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda episode_id: True)
    run_episode(env)
    env.close()
    print("Video saved to videos/ directory")
