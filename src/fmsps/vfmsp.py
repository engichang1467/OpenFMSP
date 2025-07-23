import os
import tempfile
import textwrap
import numpy as np
import importlib.util
from openai import OpenAI


class VanillaFMSP:
    def __init__(self, env_class, initial_pursuer_code, initial_evader_code, max_steps=1000, openai_model="gpt-4o-mini-2024-07-18"):
        self.env_class = env_class
        self.pursuer_code = initial_pursuer_code
        self.evader_code = initial_evader_code
        self.max_steps = max_steps
        self.openai_model = openai_model

        self.pursuer_policy = self._load_policy_from_code(self.pursuer_code, "pursuer_policy")
        self.evader_policy = self._load_policy_from_code(self.evader_code, "evader_policy")

        # Initialize attribute to store latest mean survival steps for the prompt
        self.latest_mean_n_steps = self.max_steps # Default for initial state


    def _load_policy_from_code(self, code: str, name: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
            tmp.write(code.encode("utf-8"))
            tmp_path = tmp.name

        spec = importlib.util.spec_from_file_location(name, tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        os.remove(tmp_path)
        # return mod.policy
        # Dynamically search for the policy function or class
        if hasattr(mod, 'policy'):
            return mod.policy

        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if callable(attr):
                return attr

        raise AttributeError(f"No callable policy found in module {name}.")

    
    def _generate_code_with_openai(self, prompt: str):
        api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self.openai_model, 
            messages=[
                {"role": "system", "content": "You are a helpful AI coding assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    
    def improve_policy(self, fm, role: str):
        """
        Requests the Foundation Model to improve a policy based on the latest performance.
        The prompt dynamically includes the mean survival steps for better feedback.
        """
        # Formulate the prompt to include the actual mean survival time from evaluation
        # This is critical for the FM to be "based on the latest competition result" [5] in a meaningful way.
        performance_feedback = f"The evader achieved a mean survival time of {self.latest_mean_n_steps:.2f} steps out of a maximum of {self.max_steps} steps in the recent evaluations."

        # role is either "pursuer" or "evader"
        prompt = textwrap.dedent(f"""
        Improve the following {role} policy based on the latest competition result.

        Pursuer policy:
        {self.pursuer_code}

        Evader policy:
        {self.evader_code}

        {performance_feedback}

        Please return only the improved Python code for a function named `policy(obs)`.
        Do not include any explanations, markdown formatting, triple backticks (```), or language tags like 'python'.
        Only return valid Python source code.
        """)

        if fm == "openai":
            improved_code = self._generate_code_with_openai(prompt)
            improved_code = improved_code.strip('`').replace('```python', '').replace('```', '').strip()

        else:
            raise ValueError(f"Unsupported FM: {fm}")

        try:
            new_policy = self._load_policy_from_code(improved_code, f"{role}_policy")
            # Test compilation and simple run
            self.env_class.reset()
            test_action = new_policy(np.zeros(5, dtype=np.float32))
            assert isinstance(test_action, float) or isinstance(test_action, int)
        except Exception as e:
            print(f"Failed to load new {role} policy: {e}")
            return

        if role == "pursuer":
            self.pursuer_code = improved_code
            self.pursuer_policy = new_policy
        else:
            self.evader_code = improved_code
            self.evader_policy = new_policy
    
    
    def evaluate(self, num_eval_runs=100):
        """
        Evaluates the current pursuer and evader policies over multiple runs.
        Returns the mean pursuer score, mean evader score, and mean survived steps.
        """
        pursuer_scores = []
        evader_scores = []
        n_steps_list = [] # To store n for each run

        for _ in range(num_eval_runs): # Run the simulation 100 times

            env = self.env_class
            obs, _ = env.reset()
            total_reward = 0

            n = 0 # Steps survived for the current run
            for step in range(self.max_steps):
                pursuer_action = self.pursuer_policy(obs)
                evader_action = self.evader_policy(obs)

                action = {
                    "pursuer_action": np.array([pursuer_action], dtype=np.float32),
                    "evader_action": np.array([evader_action], dtype=np.float32),
                }
                obs, _, done, _, _ = env.step(action)

                if done:
                    n = step + 1 # Evader caught, n is time survived
                    break
                n = step + 1 # If loop completes, evader survived max_steps

            evader_score_run = n / self.max_steps
            pursuer_score_run = 1 - evader_score_run

            pursuer_scores.append(pursuer_score_run)
            evader_scores.append(evader_score_run)
            n_steps_list.append(n)

        # Calculate the mean scores and mean survived steps over all runs [3]
        mean_pursuer_score = np.mean(pursuer_scores)
        mean_evader_score = np.mean(evader_scores)
        mean_n_steps = np.mean(n_steps_list)

        return mean_pursuer_score, mean_evader_score, mean_n_steps

    
    def train(self, fm, num_iterations=10, num_eval_runs=100):
        """
        Trains the vFMSP system iteratively, performing evaluations and policy improvements.
        """

        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration} ---")
            
            # Evaluate current policies over multiple runs to get mean scores
            mean_pursuer_score, mean_evader_score, mean_n_steps = self.evaluate(num_eval_runs=num_eval_runs)

            # Store the mean survival steps; this will be used by improve_policy to inform the FM
            self.latest_mean_n_steps = mean_n_steps

            # Print the mean scores for clarity and alignment with research evaluation metrics
            print(f"Mean Survived Steps: {mean_n_steps:.2f} | Mean Pursuer Win-Rate: {mean_pursuer_score:.3f} | Mean Evader Win-Rate: {mean_evader_score:.3f}")

            if iteration % 2 == 0:
                self.improve_policy(fm, role="evader")
            else:
                self.improve_policy(fm, role="pursuer")
