import random
import tqdm
import cv2
import torch
import os
import numpy as np
import gym
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
SKIP = 4
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
        initial_processed_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8) # Example
        for _ in range(NUM_STACKED_FRAMES):
            self.frame_buffer.append(initial_processed_frame)

class MemoryEfficientPERBuffer:
    """
    A Prioritized Experience Replay buffer optimized for memory efficiency.

    - Stores states as uint8 (0-255) to save memory.
    - Uses direct probability calculation for sampling (O(N) complexity).
    - Avoids the SumTree structure.
    """
    def __init__(self, capacity, input_shape=(4, 84, 84), alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.input_shape = input_shape # e.g., (4, 84, 84)
        self.alpha = alpha # Prioritization exponent
        self.beta_start = beta_start # Initial IS exponent
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.eps = 1e-6 # Small constant for priority calculation
        self.max_priority = 1.0 # Initial priority for new experiences

        # --- Data Storage ---
        # Store states compressed as uint8
        self.states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        # Store other components
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        # Store priorities directly
        self.priorities = np.zeros(capacity, dtype=np.float64) # Use float64 for precision

        # --- Pointers and Counters ---
        self.data_pointer = 0
        self.n_entries = 0 # Current number of items in buffer

        print(f"MemoryEfficientPERBuffer initialized with capacity {capacity}.")
        # Estimate memory usage (rough estimate)
        state_mem_mb = self.states.nbytes / (1024**2)
        priority_mem_mb = self.priorities.nbytes / (1024**2)
        other_mem_mb = (self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes) / (1024**2)
        print(f"Estimated memory: States ~{state_mem_mb*2:.2f} MB, Priorities ~{priority_mem_mb:.2f} MB, Others ~{other_mem_mb:.2f} MB")


    def _compress_state(self, state_float32):
        """ Converts normalized float32 state [0, 1] to uint8 [0, 255]. """
        # Ensure input is within [0, 1] before scaling
        state_clipped = np.clip(state_float32, 0.0, 1.0)
        state_uint8 = (state_clipped * 255.0).astype(np.uint8)
        return state_uint8

    def _decompress_state(self, state_uint8):
        """ Converts uint8 state [0, 255] back to normalized float32 [0, 1]. """
        return state_uint8.astype(np.float32) / 255.0

    def _get_priority(self, error):
        """ Converts TD error to priority using alpha. """
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience, compressing states to uint8.
        Assumes input states are normalized float32.
        """
        # Compress states before storing
        state_compressed = self._compress_state(state)
        next_state_compressed = self._compress_state(next_state)

        # Store data at the current pointer position
        self.states[self.data_pointer] = state_compressed
        self.actions[self.data_pointer] = action
        self.rewards[self.data_pointer] = reward
        self.next_states[self.data_pointer] = next_state_compressed
        self.dones[self.data_pointer] = done
        # Set initial priority to max
        self.priorities[self.data_pointer] = self.max_priority

        # Advance pointer and update entry count
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def sample(self, batch_size):
        """
        Samples a batch using priority probabilities and calculates IS weights.
        Decompresses states before returning. O(N) complexity.
        """
        if self.n_entries < batch_size:
            # Not enough entries to sample a full batch
            return {}, [], [] # Return empty dict, indices, weights

        # Get priorities of currently stored experiences
        current_priorities = self.priorities[:self.n_entries]

        # Calculate sampling probabilities: P(i) = p_i^alpha / sum(p_j^alpha)
        # Note: We use p_i directly here, as alpha is applied in _get_priority
        #       and stored in self.priorities (implicitly, as p = (|err|+eps)^a).
        #       So, we just need to normalize the stored priorities.
        priority_sum = np.sum(current_priorities)

        if priority_sum <= 0:
            # Avoid division by zero if all priorities are zero (should be rare with eps)
            # Fallback to uniform sampling among valid entries
            print("Priority sum is zero or negative. Falling back to uniform sampling.")
            probabilities = np.ones(self.n_entries) / self.n_entries
        else:
            probabilities = current_priorities / priority_sum

        # Sample indices based on calculated probabilities
        # Ensure probabilities sum to 1 (can have minor floating point issues)
        probabilities = probabilities / probabilities.sum()
        sampled_indices = np.random.choice(
            self.n_entries, size=batch_size, replace=True, p=probabilities
        )

        # --- Importance Sampling Weights ---
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        # Calculate IS weights: w_i = (N * P(i)) ^ (-beta)
        weights = np.power(self.n_entries * probabilities[sampled_indices], -self.beta)
        # Normalize weights by max(w) for stability
        weights /= weights.max()

        # --- Retrieve Batch Data ---
        batch_states_compressed = self.states[sampled_indices]
        batch_actions = self.actions[sampled_indices]
        batch_rewards = self.rewards[sampled_indices]
        batch_next_states_compressed = self.next_states[sampled_indices]
        batch_dones = self.dones[sampled_indices]

        # --- Decompress States ---
        batch_states = np.array([self._decompress_state(s) for s in batch_states_compressed])
        batch_next_states = np.array([self._decompress_state(ns) for ns in batch_next_states_compressed])

        batch_data = {
            "states": batch_states,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "next_states": batch_next_states,
            "dones": batch_dones # Already boolean
        }

        # Return batch, original buffer indices, and IS weights
        return batch_data, sampled_indices, weights

    def update_priorities(self, indices, errors):
        """ Updates priorities for the given indices based on TD errors. """
        if len(indices) != len(errors):
             print(f"Mismatch indices ({len(indices)}) vs errors ({len(errors)}). Skipping priority update.")
             return

        for idx, error in zip(indices, errors):
            # Ensure index is valid
            if idx < self.n_entries:
                priority = self._get_priority(error)
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
            else:
                print(f"Attempted to update priority for invalid index {idx} (n_entries={self.n_entries}).")


    def __len__(self):
        """ Returns the current number of items in the buffer. """
        return self.n_entries


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
                 gamma=0.69,           # Discount factor
                 buffer_capacity=50000,# Replay buffer capacity
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

        self.replay_buffer = MemoryEfficientPERBuffer(capacity=buffer_capacity)
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
        self.decay_epsilon()

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

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done: 
                break
        return obs, total_reward, done, info

def run_one_episode(env: JoypadSpace, agent: RainbowDQNAgent, processor: MarioPreprocessor, is_training=True):
    state = env.reset()
    processor.reset()

    done = False
    total_reward = 0
    total_steps = 0
    processed_state = processor.process(state)

    while not done:
        action = agent.select_action(processed_state, use_epsilon=is_training)
        next_state, reward, done, info = env.step(action)
        processed_next_state = processor.process(next_state)
        agent.store_experience(processed_state, action, reward, processed_next_state, done)
        processed_state = processed_next_state
        total_reward += reward
        total_steps += 1
        if total_steps % TRAIN_FREQ == 0 and is_training:
            agent.train()
    
    return total_reward

def pre_eval(env, agent, processor, iter=10):
    total_reward = 0
    for _ in range(iter):
        total_reward += run_one_episode(env, agent, processor, is_training=False)
    average_reward = total_reward / iter
    print("Pre Eval Reward: ", average_reward)
    return average_reward

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, SKIP)
    agent = RainbowDQNAgent(input_shape=(NUM_STACKED_FRAMES, FRAME_WIDTH, FRAME_HEIGHT), n_actions=12, device=DEVICE)
    processor = MarioPreprocessor()
    
    if os.path.exists(SAVE_PATH):
        agent.load_model(SAVE_PATH)

    best_reward = pre_eval(env, agent, processor, iter=5)

    for _ in tqdm.tqdm(range(TRAIN_ITER)):
        reward = run_one_episode(env, agent, processor)
        print("Reward: ", reward)
        if (_ + 1) % 100 == 0:
            average_reward = pre_eval(env, agent, processor)
            if average_reward > best_reward:
                agent.save_model(f"best_{SAVE_PATH}")
                best_reward = average_reward
            
    agent.save_model(SAVE_PATH)
    

if __name__ == "__main__":
    main()