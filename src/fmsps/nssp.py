import os
import tempfile
import textwrap
import numpy as np
import importlib.util
from openai import OpenAI # This will be mocked for the example


# --- NSSP Algorithm Implementation ---
class NSSP:
    """
    Implements the Novelty-Search Self-Play (NSSP) algorithm for open-ended strategy discovery.
    """
    def __init__(self, env_class, initial_pursuer_code: str, initial_evader_code: str,
                 max_steps_per_episode: int = 100, openai_model: str = "gpt-4o-mini-2024-07-18",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initializes the Novelty-Search Self-Play (NSSP) algorithm.
        
        Args:
            env_class: The environment class (e.g., DummyEnv).
            initial_pursuer_code (str): Python code string for the initial pursuer policy [8].
            initial_evader_code (str): Python code string for the initial evader policy [11].
            max_steps_per_episode (int): Maximum steps for a single evaluation episode [8].
            openai_model (str): The OpenAI model to use for code generation and judging [8, 10].
            embedding_model (str): The model to use for generating policy embeddings [5].
        """
        self.env_class = env_class
        self.max_steps_per_episode = max_steps_per_episode
        self.openai_model = openai_model
        self.embedding_model = embedding_model

        # Initialize archives (populations) for pursuers and evaders [3]
        self.pursuer_archive = []
        self.evader_archive = []

        # Initialize OpenAI client (or mock for demonstration)
        # In a real scenario, you'd uncomment the line below:
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        print("Initializing NSSP with baseline policies...")
        initial_pursuer_policy_obj = self._load_policy_from_code(initial_pursuer_code, "initial_pursuer_policy")
        initial_evader_policy_obj = self._load_policy_from_code(initial_evader_code, "initial_evader_policy")

        # Add initial policies to archives after checking for basic functionality [3]
        if initial_pursuer_policy_obj and self._test_policy_compilation_and_run(initial_pursuer_policy_obj, "initial_pursuer_policy"):
            self.pursuer_archive.append({
                "code": initial_pursuer_code,
                "policy_obj": initial_pursuer_policy_obj,
                "embedding": self._get_policy_embedding(initial_pursuer_code)
            })
            print(f"Added initial pursuer policy: {initial_pursuer_policy_obj.__name__}")
        else:
            print("Initial pursuer policy was buggy or failed to load. Not added.")

        if initial_evader_policy_obj and self._test_policy_compilation_and_run(initial_evader_policy_obj, "initial_evader_policy"):
            self.evader_archive.append({
                "code": initial_evader_code,
                "policy_obj": initial_evader_policy_obj,
                "embedding": self._get_policy_embedding(initial_evader_code)
            })
            print(f"Added initial evader policy: {initial_evader_policy_obj.__name__}")
        else:
            print("Initial evader policy was buggy or failed to load. Not added.")

    def _load_policy_from_code(self, code: str, name: str):
        """
        Loads a policy function or class from a Python code string [9, 11].
        Handles dynamic searching for policy objects within the module.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
            tmp.write(code.encode("utf-8"))
            tmp_path = tmp.name

        policy_obj = None
        try:
            spec = importlib.util.spec_from_file_location(name, tmp_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if hasattr(mod, 'policy'):
                policy_obj = mod.policy
                if not hasattr(policy_obj, "__name__"):
                    policy_obj.__name__ = "policy"
                return policy_obj
            else:
                # for attr_name in dir(mod):
                #     attr = getattr(mod, attr_name)
                #     if callable(attr) and not attr_name.startswith('__') and isinstance(attr, (type, type(lambda:0))):
                #         if isinstance(attr, type): # It's a class, instantiate it
                #             try: policy_obj = attr()
                #             except TypeError: continue
                #         else: policy_obj = attr # It's a function
                        
                #         if hasattr(policy_obj, '__name__') and hasattr(policy_obj, '__call__'):
                #             break
                #         elif callable(policy_obj) and hasattr(policy_obj, '__name__'):
                #             break
                #         policy_obj = None 

                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type):
                        instance = None
                        try:
                            instance = attr()
                        except Exception:
                            continue
                        if callable(instance):
                            if not hasattr(instance, "__name__"):
                                instance.__name__ = attr_name
                            return instance

                # Step 3: Fallback: search for any callable
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if callable(attr):
                        if not hasattr(attr, "__name__"):
                            attr.__name__ = attr_name
                        return attr
            
            # if policy_obj is None:
            raise AttributeError(f"No callable policy found in module {name}.")
            
            # return policy_obj
        except Exception as e:
            print(f"Error loading policy from code for {name}: {e}")
            return None
        finally:
            os.remove(tmp_path)

    def _test_policy_compilation_and_run(self, policy_obj, policy_name: str):
        """
        Performs a basic test run of a policy to check for implementation bugs [3, 12].
        """
        if policy_obj is None:
            return False
        try:
            env_test = self.env_class
            test_obs, _ = env_test.reset()
            test_action = policy_obj(test_obs)
            assert isinstance(test_action, (float, int, np.ndarray)), \
                f"Policy {policy_name} did not return valid action type."
            if isinstance(test_action, np.ndarray):
                assert test_action.ndim <= 1 and test_action.size <= 1, \
                    f"Policy {policy_name} returned an array action but expected scalar or single-element array."
            return True
        except Exception as e:
            print(f"Policy {policy_name} failed simple run test: {e}.")
            return False

    def _generate_code_with_openai(self, prompt: str):
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI coding assistant. Generate valid Python code."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7
            )
            raw_content = response.choices[0].message.content
            if raw_content.strip().startswith('```') and raw_content.strip().endswith('```'):
                content = raw_content.strip().replace('```python', '').replace('```', '').strip()
            else:
                content = raw_content.strip()
            return content
        except Exception as e:
            print(f"Failed to generate code with OpenAI: {e}")
            return None

    def _get_policy_embedding(self, code: str):
        """
        Converts policy code into an n-dimensional embedding vector [5].
        Uses a mock for demonstration.
        """
        try:
            response = self.openai_client.embeddings.create(input=code, model=self.embedding_model)
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Failed to get policy embedding: {e}")
            return np.zeros(64)

    def _find_nearest_neighbors(self, new_embedding: np.ndarray, archive: list, k: int = 5):
        """
        Finds k nearest neighbors in the archive based on embedding distance [5].
        """
        if not archive:
            return []

        distances = []
        for i, policy_info in enumerate(archive):
            current_embedding = np.array(policy_info["embedding"])
            dist = np.linalg.norm(new_embedding - current_embedding)
            distances.append((dist, i))

        distances.sort(key=lambda x: x)
        nearest_neighbors_info = [archive[idx] for dist, idx in distances[:k]]
        return nearest_neighbors_info

    def _fm_as_judge_novelty(self, new_policy_code: str, neighbors_info: list):
        """
        Uses the FM as a judge to determine if the new policy is truly novel [6, 7].
        This replaces the density-based analysis from traditional novelty search [7].
        """
        neighbor_codes_str = ""
        if neighbors_info:
            for i, n in enumerate(neighbors_info):
                neighbor_codes_str += textwrap.dedent(f"""
                Policy {i+1} Name: {n['policy_obj'].__name__}
                Policy {i+1} Code:
                {textwrap.indent(n['code'], '    ')}
                """)
        else:
            neighbor_codes_str = "No existing neighbors to compare against (archive is small or empty)."

        prompt = textwrap.dedent(f"""
        You are an expert judge of code novelty. Your task is to determine if a NEW policy is
        "interestingly new" compared to a set of EXISTING neighboring policies.
        Consider its conceptual approach, algorithms used, and overall behavior, not just minor syntax changes.
        The goal is to foster pure exploration and diversity, so even slight conceptual shifts are valued.

        NEW Policy Code:
        {new_policy_code}

        EXISTING Neighboring Policies (ordered by similarity in embedding space):
        {neighbor_codes_str}

        Is the NEW Policy truly novel and distinct enough from these neighboring policies to be added to an archive focused on diversity?
        Respond concisely with "Yes" or "No". Do not include any other explanations, markdown formatting, or text.
        """)

        try:
            response_content = self._generate_code_with_openai(prompt) 
            return "yes" in response_content.lower()
        except Exception as e:
            print(f"Failed to query FM-as-judge for novelty: {e}")
            return False 

    def _evaluate_policies_head_to_head(self, pursuer_policy_obj, evader_policy_obj):
        """
        Simulates a head-to-head competition to get a performance score for FM context [3].
        Returns the evader's mean survival steps.
        """
        num_runs = 5 # Small number of runs for context, not full evaluation
        n_steps_list = []

        for _ in range(num_runs):
            env = self.env_class
            obs, _ = env.reset()
            n = 0 
            for step in range(self.max_steps_per_episode):
                try:
                    pursuer_action = pursuer_policy_obj(obs)
                    evader_action = evader_policy_obj(obs)
                    
                    action_dict = {
                        "pursuer_action": np.array([pursuer_action], dtype=np.float32),
                        "evader_action": np.array([evader_action], dtype=np.float32),
                    }
                    
                    obs, _, done, _, _ = env.step(action_dict)
                    if done:
                        n = step + 1 
                        break
                    n = step + 1 
                except Exception as e:
                    print(f"Policy evaluation error in head-to-head: {e}. Assuming minimal survival.")
                    n = 1 
                    break
            n_steps_list.append(n)
        
        return np.mean(n_steps_list) 

    def _select_policies_for_context(self, current_archive: list, opponent_archive: list, role: str):
        """
        Selects policies (and their code) from archives to provide as context to the FM [3].
        """
        # Fallback to dummy policies if archives are unexpectedly empty
        # dummy_policy_info = {"code": "class Dummy: def __init__(self): self.__name__='Dummy'; self.description='Dummy';\n def __call__(self, X): return 0.0", 
        #                      "policy_obj": self._load_policy_from_code("class Dummy: def __init__(self): self.__name__='Dummy'; self.description='Dummy';\n def __call__(self, X): return 0.0", "Dummy"), 
        #                      "embedding": np.zeros(64)}

        selected_current_policy = current_archive[np.random.randint(len(current_archive))] if current_archive else None # dummy_policy_info
        selected_opponent_policy = opponent_archive[np.random.randint(len(opponent_archive))] if opponent_archive else None # dummy_policy_info

        # Get head-to-head performance [3]
        # if selected_current_policy == dummy_policy_info or selected_opponent_policy == dummy_policy_info:
        #     head_to_head_score = 0
        # el
        if role == "pursuer":
            head_to_head_score = self._evaluate_policies_head_to_head(
                selected_current_policy["policy_obj"], selected_opponent_policy["policy_obj"]
            )
        else: # evader
            head_to_head_score = self._evaluate_policies_head_to_head(
                selected_opponent_policy["policy_obj"], selected_current_policy["policy_obj"]
            )

        # Include neighboring policies from the current agent's own archive [3]
        temp_archive_for_neighbors = [p for p in current_archive if p != selected_current_policy]
        neighbors_raw = self._find_nearest_neighbors(
            selected_current_policy["embedding"], temp_archive_for_neighbors, k=3
        )
        # Format neighbors for context prompt
        neighbors_for_context = [{"name": n_info["policy_obj"].__name__, "code": n_info["code"]} for n_info in neighbors_raw]

        return selected_current_policy, selected_opponent_policy, head_to_head_score, neighbors_for_context

    def generate_and_add_policy(self, role: str):
        """
        Executes a single step of the NSSP algorithm: generating, testing, and archiving a new policy [3, 5, 6, 14].
        """
        print(f"\n--- Attempting to generate and add a new {role} policy ---")
        
        # Determine current and opponent archives based on the role
        if role == "pursuer":
            current_archive = self.pursuer_archive
            opponent_archive = self.evader_archive
            agent_type_full = "pursuer"
            opponent_type_full = "evader"
        elif role == "evader":
            current_archive = self.evader_archive
            opponent_archive = self.pursuer_archive
            agent_type_full = "evader"
            opponent_type_full = "pursuer"
        else:
            raise ValueError("Role must be 'pursuer' or 'evader'.")

        # Step 1: Select policies for context to the FM for new policy generation [3]
        selected_current_policy_info, selected_opponent_policy_info, head_to_head_score, neighbors_for_context = \
            self._select_policies_for_context(current_archive, opponent_archive, role)

        # Construct the context string for the FM's prompt
        context_str = textwrap.dedent(f"""
        Current {agent_type_full} policy (randomly sampled from archive):
        Name: {selected_current_policy_info['policy_obj'].__name__}
        Code:
        {textwrap.indent(selected_current_policy_info['code'], '    ')}

        Opponent ({opponent_type_full}) policy (randomly sampled from archive):
        Name: {selected_opponent_policy_info['policy_obj'].__name__}
        Code:
        {textwrap.indent(selected_opponent_policy_info['code'], '    ')}

        Head-to-head performance of the current {agent_type_full} vs. the opponent {opponent_type_full} (evader survival steps): {head_to_head_score:.2f} out of {self.max_steps_per_episode} steps.
        """)
        
        if neighbors_for_context:
            context_str += "\nNeighboring policies from the same archive (to ensure distinctness):\n"
            for i, n_info in enumerate(neighbors_for_context):
                context_str += textwrap.dedent(f"""
                Neighbor {i+1} Name: {n_info['name']}
                Code:
                {textwrap.indent(n_info['code'], '    ')}
                """)
        else:
            context_str += "\nNo additional neighbors for context (archive too small).\n"

        # The core prompt instructing the FM to generate a *distinct* policy, ignoring performance [3]
        prompt_template = textwrap.dedent(f"""
        You are an expert at designing novel policies that drive multi-agent innovation.
        Based on the provided context, generate a new and **distinct** {agent_type_full} policy.
        The primary goal is to foster **pure exploration and diversity**, explicitly ignoring performance considerations.
        Focus on creating a unique conceptual approach or strategy, even if it is not immediately performant.
        Do not make something similar to the policies provided in the context.

        Context:
        {context_str}

        Please return only the improved Python code for a policy class (e.g., `class MyNewPolicy:`).
        Do not include any explanations, markdown formatting, triple backticks (```), or language tags like 'python'.
        Only return valid Python source code. The class name should be unique and descriptive.
        """)

        # Step 2: FM generates new policy code [3]
        new_policy_code = self._generate_code_with_openai(prompt_template)
        if not new_policy_code:
            print(f"Failed to generate new {role} code. Skipping this iteration.")
            return

        # Step 3: Refine and test for implementation bugs [3]
        new_policy_obj = self._load_policy_from_code(new_policy_code, f"generated_{role}_policy_{len(current_archive)}")
        if not new_policy_obj:
            print(f"New {role} policy has compilation errors. Rejected.")
            return
        
        if not self._test_policy_compilation_and_run(new_policy_obj, new_policy_obj.__name__):
            print(f"New {role} policy ({new_policy_obj.__name__}) failed simple run test. Rejected.")
            return

        # Step 4: Convert to embedding vector [5]
        new_policy_embedding = self._get_policy_embedding(new_policy_code)

        # Step 5: Find k nearest neighbors in the current archive for novelty check [5]
        neighbors_in_archive = self._find_nearest_neighbors(new_policy_embedding, current_archive, k=3)

        # Step 6: Query FM-as-judge for novelty [6, 7]
        is_novel = self._fm_as_judge_novelty(new_policy_code, neighbors_in_archive)

        if is_novel:
            # Step 7: If novel, add to archive (ignoring performance) [6]
            current_archive.append({
                "code": new_policy_code,
                "policy_obj": new_policy_obj,
                "embedding": new_policy_embedding
            })
            print(f"--- Successfully added novel {role} policy to archive: {new_policy_obj.__name__} (Archive size: {len(current_archive)}) ---")
        else:
            print(f"--- New {role} policy ({new_policy_obj.__name__}) was not considered novel by FM-as-judge. Rejected. ---")

    def train(self, num_iterations: int = 10):
        """
        Runs the NSSP training loop for a specified number of iterations, 
        alternating between generating new evader and pursuer policies.
        """
        print(f"Starting Novelty-Search Self-Play (NSSP) training for {num_iterations} iterations...")
        for iteration in range(num_iterations):
            print(f"\n======== NSSP Iteration {iteration + 1}/{num_iterations} ========")
            
            # Alternate policy generation between evader and pursuer
            if iteration % 2 == 0:
                self.generate_and_add_policy(role="evader")
            else:
                self.generate_and_add_policy(role="pursuer")
            
            print(f"Iteration {iteration + 1} Summary:")
            print(f"  Pursuer Archive size: {len(self.pursuer_archive)}")
            print(f"  Evader Archive size: {len(self.evader_archive)}")

        print(f"\nNSSP training complete after {num_iterations} iterations.")
        print(f"Final Pursuer Archive size: {len(self.pursuer_archive)}")
        print(f"Final Evader Archive size: {len(self.evader_archive)}")
