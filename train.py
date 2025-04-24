import random
import tqdm
import cv2
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch import nn

TRAIN_ITER = 10

class MarioPreprocessor:
    """
    Handles preprocessing of single game frames for Super Mario Bros.
    - Converts RGB frames to grayscale.
    - Resizes frames to a specified output size.
    - Normalizes pixel values to the range [0, 1].
    """
    def __init__(self, output_size=(84, 84)):
        """
        Initializes the preprocessor.

        Args:
            output_size (tuple): The desired output size (height, width)
                                 for the processed frames. Defaults to (84, 84).
        """
        # Ensure output_size is in (width, height) format for cv2.resize
        # but store internally maybe as (height, width) for clarity
        self.output_height = output_size[0]
        self.output_width = output_size[1]
        # cv2 resize expects (width, height)
        self.cv2_output_size = (self.output_width, self.output_height)
        print(f"Preprocessor initialized for output size: (Height: {self.output_height}, Width: {self.output_width})")

    def process(self, frame):
        """
        Processes a single raw game frame.

        Args:
            frame (np.ndarray): The raw input frame from the environment,
                                expected shape (H, W, 3) with RGB channels
                                and dtype uint8.

        Returns:
            np.ndarray: The processed frame, shape (output_height, output_width),
                        dtype float32, with pixel values normalized to [0, 1].
                        Returns None if the input frame is invalid.
        """
        # Basic validation: Check if it's a 3-channel image
        if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
            print("Warning: Invalid frame received for processing.")
            return None # Or raise an error

        # 1. Convert frame to Grayscale
        # cv2.COLOR_RGB2GRAY assumes input is RGB. If the env provides BGR, use cv2.COLOR_BGR2GRAY
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Result shape: (H, W)

        # 2. Resize frame to output_size
        # cv2.INTER_AREA is generally recommended for downsampling (shrinking images)
        resized_frame = cv2.resize(
            gray_frame,
            self.cv2_output_size, # cv2 expects (width, height)
            interpolation=cv2.INTER_AREA
        )
        # Result shape: (output_height, output_width)

        # 3. Normalize pixel values to [0, 1] and change dtype
        # Ensure the frame is float before division
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        # Add channel dimension if needed downstream (e.g., for PyTorch Conv2d expecting C, H, W)
        # Usually FrameStack handles the channel dimension, but if using single frames:
        # normalized_frame = np.expand_dims(normalized_frame, axis=0) # Shape: (1, H, W)
        return normalized_frame

class SumTree:
    """
    A binary tree data structure where the value of a parent node
    is the sum of its children. Leaf nodes store priorities, and
    internal nodes store sums. Allows O(log N) operations for
    sampling and updates.
    """
    def __init__(self, capacity):
        # Tree structure: Stores priorities and sums. Size is 2*capacity - 1.
        # Index 0 is the root. Leaves start at index capacity - 1.
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # Data storage: Stores the actual experience tuples.
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0 # Points to the next empty spot in self.data
        self.n_entries = 0 # Current number of entries in the buffer

    def _propagate(self, idx, change):
        """Propagates a change in priority up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: # If not root
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Finds the sample index for a given priority value 's'."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree): # Reached leaf node
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Returns the total priority sum (value of the root node)."""
        return self.tree[0]

    def add(self, priority, data):
        """Adds a new experience with a given priority."""
        # Calculate the tree index for the new data point
        tree_idx = self.data_pointer + self.capacity - 1

        # Store the data
        self.data[self.data_pointer] = data

        # Update the priority in the tree
        self.update(tree_idx, priority)

        # Advance the data pointer (cyclical)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        # Update the count of entries, capped at capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """Updates the priority of a node and propagates the change."""
        if not (self.capacity - 1 <= tree_idx < 2 * self.capacity - 1):
             raise IndexError(f"tree_idx {tree_idx} out of bounds for leaves [{self.capacity - 1}, {2 * self.capacity - 1})")

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate the change upwards
        self._propagate(tree_idx, change)

    def get(self, s):
        """
        Gets the leaf index, priority, and data for a given sampled
        priority value 's'.
        """
        idx = self._retrieve(0, s) # Get tree index of the leaf
        data_idx = idx - self.capacity + 1 # Convert tree index to data index
        return (idx, self.tree[idx], self.data[data_idx])

    def __len__(self):
        return self.n_entries

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.sum_tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha # Controls prioritization level (0=uniform, 1=full)
        self.beta_start = beta_start # Initial IS weight exponent
        self.beta = beta_start # Current IS weight exponent (annealed externally)
        self.beta_increment_per_sampling = (1.0 - beta_start) / beta_frames
        self.eps = 1e-6  # Small value added to priorities to ensure non-zero probability
        self.max_priority = 1.0 # Initial max priority for new samples

    def _get_priority(self, error):
        """Converts TD error to priority."""
        # P = (|error| + eps) ^ alpha
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the buffer with maximum priority initially.
        Assigning max priority ensures new experiences are likely to be sampled soon.
        """
        # Store experience in SumTree with current max priority
        priority = self.max_priority
        self.sum_tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a batch of experiences based on priorities and calculates
        Importance Sampling (IS) weights.
        """
        batch = []
        idxs = [] # Store tree indices
        segment = self.sum_tree.total() / batch_size
        priorities = []

        # Anneal beta
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        if len(self) == 0: # Check if buffer is empty
             return [], [], [] # Return empty lists if no samples

        for i in range(batch_size):
            # Sample a value from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # Retrieve data associated with the sampled value
            (idx, p, data) = self.sum_tree.get(s)

            if data == 0: # Check if data is placeholder (can happen if buffer not full)
                 # Resample if we hit an empty spot (should be rare with proper n_entries logic)
                 # This indicates an issue, maybe buffer isn't filling correctly or total() is wrong
                 print(f"Warning: Sampled empty data slot at tree_idx {idx}. Resampling.")
                 # Simple fix: retry sampling (could be inefficient)
                 # A better fix might involve checking n_entries vs capacity
                 s_retry = random.uniform(0, self.sum_tree.total())
                 (idx, p, data) = self.sum_tree.get(s_retry)
                 if data == 0:
                     print(f"Error: Failed to sample valid data even after retry.")
                     # Handle error appropriately, maybe return smaller batch or raise exception
                     continue # Skip this sample


            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Calculate sampling probabilities P(i) = p_i / total_priority
        sampling_probabilities = np.array(priorities) / self.sum_tree.total()

        # Calculate Importance Sampling (IS) weights w_i = (N * P(i)) ^ (-beta) / max(w)
        # Use self.sum_tree.n_entries for N (current number of items)
        is_weights = np.power(self.sum_tree.n_entries * sampling_probabilities, -self.beta)

        # Normalize weights by dividing by the maximum weight for stability
        if is_weights.size > 0: # Avoid division by zero if batch is empty
             is_weights /= is_weights.max()
        else:
             is_weights = np.array([])


        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        batch_data = {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32) # Ensure float for later math
        }

        return batch_data, idxs, is_weights # Return batch, tree indices, and IS weights

    def update_priorities(self, tree_indices, errors):
        """
        Updates the priorities of the sampled experiences based on their TD errors.
        """
        if len(tree_indices) != len(errors):
            raise ValueError("Number of indices and errors must match.")

        for idx, error in zip(tree_indices, errors):
            # Calculate new priority based on error
            priority = self._get_priority(error)
            # Update the priority in the SumTree
            self.sum_tree.update(idx, priority)
            # Update the maximum priority observed so far
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """Returns the current number of items in the buffer."""
        return self.sum_tree.n_entries

class DuelingQNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DuelingQNet, self).__init__()
        self.n_actions = n_actions

        # Shared feature layers (first part of the original network)
        self.feature_layer = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
            # Output of this layer has size 64
        )

        # Value stream head - estimates V(s)
        # Takes the 64 features and processes them further
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64), # Continues from the shared 64 features
            nn.ReLU(),
            nn.Linear(64, 1)   # Outputs a single scalar value V(s)
        )

        # Advantage stream head - estimates A(s, a) for each action
        # Also takes the 64 features
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64), # Continues from the shared 64 features
            nn.ReLU(),
            nn.Linear(64, n_actions) # Outputs one advantage value per action A(s,a)
        )

    def forward(self, x):
        # Pass input through the shared feature layers
        features = self.feature_layer(x) # Shape: (batch_size, 64)

        # Calculate Value and Advantage streams
        value = self.value_stream(features)           # Shape: (batch_size, 1)
        advantages = self.advantage_stream(features)   # Shape: (batch_size, n_actions)

        # Combine Value and Advantage streams to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Subtracting the mean advantage is crucial for identifiability and stability
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        # value has shape (batch, 1), advantages has shape (batch, n_actions)
        # advantages.mean(dim=1, keepdim=True) has shape (batch, 1)
        # Broadcasting correctly combines them to shape (batch, n_actions)

        return q_values

class RainbowDQNAgent:
    def __init__(self, env):
        self.action_space = env.action_space

    def select_action(self, state):
        print(state.shape)
        return self.action_space.sample()

def run_one_episode(env: JoypadSpace, agent: RainbowDQNAgent, processor: MarioPreprocessor):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        processed_state = processor.process(state)
        action = agent.select_action(processed_state)
        state, reward, done, info = env.step(action)
        total_reward += reward
    
    return total_reward

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    agent = RainbowDQNAgent(env=env)
    processor = MarioPreprocessor()
    
    for _ in tqdm.tqdm(range(TRAIN_ITER)):
        reward = run_one_episode(env, agent, processor)
        print("Reward: ", reward)

if __name__ == "__main__":
    main()