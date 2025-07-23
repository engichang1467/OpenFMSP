from src.envs.car_tag.env import CarTagEnv
from src.fmsps.vfmsp import VanillaFMSP

initial_pursuer_code = """
import numpy as np

# Define a constant for tuning the pursuer's turning rate.
# This value can be adjusted to change the responsiveness of the pursuer.
PURSUER_TURN_CONSTANT = 0.1

def policy(obs):

    # Pursuer policy that calculates a turning action (phi) based on the
    # relative position of the evader and the pursuer's current heading.

    # Args:
    #     obs (np.array): The current observation from the environment,
    #                     structured as [xp, yp, theta, xe, ye], where:
    #                     xp, yp: Pursuer's x and y coordinates
    #                     theta: Pursuer's heading angle
    #                     xe, ye: Evader's x and y coordinates

    # Returns:
    #     float: The turning rate ratio (phi) for the pursuer,
    #            a value between -1.0 and 1.0.

    xp, yp, theta, xe, ye = obs

    # Calculate the angle from the pursuer's current position to the evader's position.
    # np.arctan2(dy, dx) gives the angle in radians between the positive x-axis
    # and the point (dx, dy).
    angle_to_evader = np.arctan2(ye - yp, xe - xp)

    # Calculate the difference between the pursuer's current heading (theta)
    # and the desired heading towards the evader.
    # The result is wrapped to be within -pi to pi to ensure the shortest turn direction.
    angle_difference = angle_to_evader - theta
    angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))

    # Determine the turning action (phi).
    # This is a proportional control: the larger the angle difference, the stronger the turn.
    # The PURSUER_TURN_CONSTANT scales this response.
    # The action is clipped to ensure it stays within the valid range of -1.0 to 1.0,
    # as defined by the environment's action space for the pursuer.
    phi = np.clip(angle_difference / PURSUER_TURN_CONSTANT, -1.0, 1.0)

    return phi
"""

# Modified initial evader policy: constant heading straight line
initial_evader_code = """
import numpy as np

# Define a constant for the evader's random turning magnitude.
# This can be adjusted to make the evader more or less erratic.
EVADER_RANDOM_TURN_MAGNITUDE = np.pi / 4 # e.g., turn up to 45 degrees randomly

def policy(obs):

    # Evader policy that generates a random heading angle (psi).
    # This policy causes the evader to move in a somewhat unpredictable direction.

    # Args:
    #     obs (np.array): The current observation from the environment.
    #                     (Note: The evader policy currently does not use the observation,
    #                     but it's included for consistency with the expected function signature.)

    # Returns:
    #     float: A random heading angle (psi) in radians,
    #            a value between -np.pi and np.pi.

    # Generate a random angle for the evader's heading.
    # np.random.uniform(low, high) generates a random float within the specified range.
    # The range -np.pi to np.pi covers all possible angles for a heading.
    psi = np.random.uniform(-np.pi, np.pi)

    return psi
"""


if __name__ == "__main__":

    env = CarTagEnv(render_mode="rgb_array")

    # Create vFMSP system
    vfms = VanillaFMSP(
        env_class=env,
        initial_pursuer_code=initial_pursuer_code,
        initial_evader_code=initial_evader_code,
        max_steps=300,
        openai_model="gpt-4o-mini-2024-07-18"
    )

    # Train the system
    vfms.train(fm="openai", num_iterations=10, num_eval_runs=100)

    # Final evaluation
    print("\nFinal Evaluation:")
    mean_p_score, mean_e_score, mean_steps = vfms.evaluate(num_eval_runs=100)
    print(f"Mean Pursuer Score: {mean_p_score:.3f}")
    print(f"Mean Evader Score: {mean_e_score:.3f}")
    print(f"Mean Survived Steps: {mean_steps:.2f}")
