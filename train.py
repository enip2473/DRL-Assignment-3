import random
import tqdm
import cv2
import torch
import os
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch import nn
import torch.optim as optim
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_STACKED_FRAMES = 4
FRAME_HEIGHT = 84
FRAME_WIDTH = 84
TRAIN_ITER = 10000
TRAIN_FREQ = 4
SAVE_PATH = "model_weight.pth"

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
        self.cv2_output_size = (self.output_width, self.output_height)
        self.frame_buffer = deque(maxlen=NUM_STACKED_FRAMES)
        initial_processed_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8) # Example
        for _ in range(NUM_STACKED_FRAMES):
            self.frame_buffer.append(initial_processed_frame)

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
        self.frame_buffer.append(resized_frame)
        # 3. Normalize pixel values to [0, 1] and change dtype
        # Ensure the frame is float before division
        stacked_frames = np.stack(self.frame_buffer, axis=0) 
        # Add channel dimension if needed downstream (e.g., for PyTorch Conv2d expecting C, H, W)
        # Usually FrameStack handles the channel dimension, but if using single frames:
        normalized_state = stacked_frames.astype(np.float32) / 255.0
        return normalized_state
    
    def reset(self):
        initial_processed_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32) # Example
        for _ in range(NUM_STACKED_FRAMES):
            self.frame_buffer.append(initial_processed_frame)

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

class DuelingCNNQNet(nn.Module):
    """
    Dueling Q-Network with CNN feature extractor, suitable for image inputs.
    """
    def __init__(self, input_shape, n_actions):
        """
        Args:
            input_shape (tuple): Shape of the preprocessed input state
                                 (e.g., (4, 84, 84) for 4 stacked 84x84 frames).
            n_actions (int): Number of possible actions.
        """
        super(DuelingCNNQNet, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        # --- Convolutional Feature Extractor ---
        # Takes input shape (Channels, Height, Width), e.g., (4, 84, 84)
        self.conv_layers = nn.Sequential(
            # Conv1: Input (4, 84, 84) -> Output (32, 20, 20)
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Conv2: Input (32, 20, 20) -> Output (64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Conv3: Input (64, 9, 9) -> Output (64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            # Output features have shape (BatchSize, 64, 7, 7)
        )

        # --- Calculate flattened size ---
        # We need to know the output size of conv_layers to define the Linear layers
        conv_out_size = self._get_conv_output_size(input_shape)

        # --- Dueling Heads ---

        # Value Stream Head - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512), # Takes flattened features
            nn.ReLU(),
            nn.Linear(512, 1)              # Outputs a single scalar value V(s)
        )

        # Advantage Stream Head - estimates A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512), # Takes flattened features
            nn.ReLU(),
            nn.Linear(512, n_actions)      # Outputs one advantage value per action A(s,a)
        )

    def _get_conv_output_size(self, shape):
        """
        Calculates the output size of the convolutional layers by performing
        a dummy forward pass.
        """
        # Create a dummy tensor with the expected input shape (adding batch dim)
        dummy_input = torch.zeros(1, *shape)
        # Pass it through the convolutional layers
        output = self.conv_layers(dummy_input)
        # Calculate the total number of features after flattening
        return int(np.prod(output.size()))

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor (preprocessed states).
                              Shape: (batch_size, C, H, W), e.g., (32, 4, 84, 84).
                              Ensure pixel values are normalized (e.g., scaled to [0, 1]).
        Returns:
            torch.Tensor: Q-values for each action. Shape: (batch_size, n_actions).
        """
        # Normalize input if not already done (e.g., assuming input is 0-255)
        # It's often better to do this in preprocessing/data loading
        # x = x / 255.0

        # Pass input through the convolutional feature extractor
        conv_features = self.conv_layers(x) # Shape: (batch_size, 64, 7, 7)

        # Flatten the features for the fully connected layers
        # Shape becomes (batch_size, 64 * 7 * 7) = (batch_size, 3136)
        flattened_features = conv_features.view(conv_features.size(0), -1)

        # Calculate Value and Advantage streams
        value = self.value_stream(flattened_features)      # Shape: (batch_size, 1)
        advantages = self.advantage_stream(flattened_features) # Shape: (batch_size, n_actions)

        # Combine Value and Advantage streams using the Dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

class RainbowDQNAgent:
    """
    DQN Agent integrating Dueling architecture, Double DQN, and Prioritized Experience Replay.
    Designed for image-based environments like Super Mario Bros.
    Note: This is a partial Rainbow implementation (missing N-step, C51, Noisy Nets).
    """
    def __init__(self,
                 input_shape,          # Shape of preprocessed state (e.g., (4, 84, 84))
                 n_actions,            # Number of possible actions
                 device,               # Device to run on ('cpu' or 'cuda')
                 lr=1e-4,              # Learning rate
                 gamma=0.99,           # Discount factor
                 buffer_capacity=100000,# Replay buffer capacity
                 batch_size=32,        # Training batch size
                 target_update_freq=1000,# How often to update target network (in steps)
                 epsilon_start=1.0,    # Initial exploration rate
                 epsilon_min=0.01,     # Minimum exploration rate
                 epsilon_decay=0.9999): # Epsilon decay factor per step/call

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0 # Counter for steps taken

        print(f"Initializing Agent on device: {self.device}")
        print(f"Input shape: {input_shape}, Actions: {n_actions}")

        # --- Networks ---
        print("Creating Q-Network and Target Network...")
        self.q_net = DuelingCNNQNet(input_shape, n_actions).to(self.device)
        self.target_net = DuelingCNNQNet(input_shape, n_actions).to(self.device)
        # Synchronize target network initially
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Target network is only for inference
        print("Networks created and target network synchronized.")

        # --- Optimizer ---
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        print(f"Optimizer Adam initialized with lr={lr}")

        # --- Replay Buffer ---
        print(f"Initializing Prioritized Replay Buffer with capacity {buffer_capacity}...")
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        print("Replay Buffer initialized.")
        # --- Loss Function ---
        # Using MSE Loss here, but SmoothL1Loss is also common
        # We'll apply IS weights manually
        self.loss_fn = nn.MSELoss(reduction='none') # Calculate element-wise loss


    def select_action(self, state, use_epsilon=True):
        """
        Selects an action using epsilon-greedy strategy.

        Args:
            state (np.ndarray): The *preprocessed* input state (e.g., shape (4, 84, 84)).
            use_epsilon (bool): Whether to apply epsilon-greedy exploration.

        Returns:
            int: The selected action index.
        """
        # Epsilon-greedy exploration
        if use_epsilon and random.random() <= self.epsilon:
            return random.randrange(self.n_actions)
        else:
            # Convert state to tensor, add batch dimension, send to device
            # Ensure state is float32
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Get Q-values from the main network
            self.q_net.eval() # Set to evaluation mode for inference
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            self.q_net.train() # Set back to training mode

            # Select action with the highest Q-value
            action = torch.argmax(q_values).item()
            return action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay buffer.
        Assumes states are already preprocessed.

        Args:
            state (np.ndarray): Preprocessed current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Preprocessed next state.
            done (bool): Whether the episode terminated.
        """
        # Add experience to PER buffer with max priority initially
        self.replay_buffer.add(state, action, reward, next_state, done)

    def _update_target_network(self):
        """Copies weights from the main Q-network to the target network."""
        print(f"\n--- Step {self.total_steps}: Updating target network ---")
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _compute_td_error_and_loss(self, batch_data, indices, is_weights):
        """
        Computes the TD errors and the weighted loss for a batch using Double DQN.

        Args:
            batch_data (dict): Dictionary containing 'states', 'actions', etc. as np.arrays.
            indices (list): List of tree indices for the sampled transitions.
            is_weights (np.ndarray): Importance sampling weights for the batch.

        Returns:
            tuple: (loss (torch.Tensor), td_errors (np.ndarray))
        """
        # Convert numpy arrays from batch_data to tensors on the correct device
        states = torch.tensor(batch_data['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch_data['actions'], dtype=torch.int64).unsqueeze(1).to(self.device) # Shape: (batch, 1)
        rewards = torch.tensor(batch_data['rewards'], dtype=torch.float32).unsqueeze(1).to(self.device) # Shape: (batch, 1)
        next_states = torch.tensor(batch_data['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch_data['dones'], dtype=torch.float32).unsqueeze(1).to(self.device) # Shape: (batch, 1)
        is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1).to(self.device) # Shape: (batch, 1)

        # --- Double DQN Logic ---
        # 1. Get Q-values for next states from the main network
        self.q_net.eval() # Use eval mode for action selection consistency
        with torch.no_grad():
             q_values_next_main = self.q_net(next_states)
        self.q_net.train() # Back to train mode

        # 2. Select the best actions for next states using the main network's Q-values
        # Shape: (batch, 1)
        best_actions_next = torch.argmax(q_values_next_main, dim=1, keepdim=True)

        # 3. Evaluate these selected actions using the target network
        # Shape: (batch, 1)
        with torch.no_grad():
            q_values_next_target = self.target_net(next_states).gather(1, best_actions_next)
        # --- End Double DQN ---

        # Calculate TD Target
        # target = r + gamma * Q_target(s', argmax_a' Q_main(s', a')) * (1 - done)
        td_target = rewards + self.gamma * q_values_next_target * (1 - dones)

        # Get Q-values for the current states and actions taken (from main network)
        # Shape: (batch, 1)
        current_q_values = self.q_net(states).gather(1, actions)

        # Calculate element-wise loss (e.g., MSE or Smooth L1)
        elementwise_loss = self.loss_fn(current_q_values, td_target)

        # Apply Importance Sampling weights
        weighted_loss = elementwise_loss * is_weights_tensor

        # Calculate the final loss (mean over the batch)
        loss = weighted_loss.mean()

        # Calculate TD errors (absolute difference) for priority updates
        # Use .detach() to prevent gradients from flowing back from this calculation
        td_errors = (td_target - current_q_values).abs().detach().cpu().numpy().flatten()

        return loss, td_errors

    def train(self):
        """
        Samples a batch from the replay buffer, computes loss, performs backprop,
        updates network weights, and updates priorities in the buffer.
        Also handles periodic target network updates.
        """
        # Only train if buffer has enough samples and meets batch size requirement
        if len(self.replay_buffer) < self.batch_size:
            # print(f"Skipping training step {self.total_steps}. Buffer size {len(self.replay_buffer)} < Batch size {self.batch_size}")
            return # Not enough samples yet

        # Sample batch from PER buffer
        batch_data, indices, is_weights = self.replay_buffer.sample(self.batch_size)

        if not batch_data: # Handle case where sampling might fail (e.g., empty buffer)
             print("Warning: Sampled empty batch. Skipping training step.")
             return

        # Compute loss and TD errors
        loss, td_errors = self._compute_td_error_and_loss(batch_data, indices, is_weights)

        # --- Gradient Descent ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        # --- End Gradient Descent ---

        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)

        # Increment step counter (used for target updates and epsilon decay)
        self.total_steps += 1

        # --- Periodic Target Network Update ---
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()

        # --- Epsilon Decay ---
        # Decay epsilon after each training step or episode end (choose one)
        # Decaying per step is common in large-scale experiments
        self.decay_epsilon()

        # Return loss value for logging if needed
        return loss.item()


    def decay_epsilon(self):
        """Decays the exploration rate epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        """Saves the Q-network weights."""
        print(f"\nSaving model Q-network state_dict to {path}...")
        torch.save(self.q_net.state_dict(), path)
        print("Model saved.")

    def load_model(self, path):
        """Loads the Q-network weights."""
        print(f"\nLoading model Q-network state_dict from {path}...")
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        # Also update target network to match loaded weights
        self._update_target_network()
        self.q_net.train() # Ensure model is in train mode after loading
        self.target_net.eval() # Ensure target is in eval mode
        print("Model loaded and target network synchronized.")

def run_one_episode(env: JoypadSpace, agent: RainbowDQNAgent, processor: MarioPreprocessor):
    state = env.reset()
    processor.reset()

    done = False
    total_reward = 0
    total_steps = 0
    processed_state = processor.process(state)
    while not done:
        action = agent.select_action(processed_state)
        next_state, reward, done, info = env.step(action)
        processed_next_state = processor.process(next_state)
        agent.store_experience(processed_state, action, reward, processed_next_state, done)
        processed_state = processed_next_state
        total_reward += reward
        total_steps += 1
        if total_steps % TRAIN_FREQ == 0:
            agent.train()
    
    return total_reward

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    agent = RainbowDQNAgent(input_shape=(NUM_STACKED_FRAMES, FRAME_WIDTH, FRAME_HEIGHT), n_actions=12, device=DEVICE)
    processor = MarioPreprocessor()
    
    if os.path.exists(SAVE_PATH):
        agent.load_model(SAVE_PATH)

    for _ in tqdm.tqdm(range(TRAIN_ITER)):
        reward = run_one_episode(env, agent, processor)
        print("Reward: ", reward)
        if (_ + 1) % 100 == 0:
            agent.save_model(f"tmp_{SAVE_PATH}")
            
    agent.save_model(SAVE_PATH)
    

if __name__ == "__main__":
    main()