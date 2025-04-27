import gym
from train import RainbowDQNAgent, MarioPreprocessor
import torch
# Do not modify the input of the 'act' function and the '__init__' function. 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_model_weight.pth"

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.agent = RainbowDQNAgent((4, 84, 84), 12, DEVICE)
        self.agent.load_model(SAVE_PATH)
        self.processor = MarioPreprocessor()
        self.last_action = None
        self.skip_count = 0
    
    def act(self, observation):
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action
        processed_state = self.processor.process(observation)
        action = self.agent.select_action(processed_state, use_epsilon=False)
        self.skip_count = 4
        self.last_action = action
        return action