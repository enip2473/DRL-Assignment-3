import gym
from train import RainbowDQNAgent, MarioPreprocessor
import torch
# Do not modify the input of the 'act' function and the '__init__' function. 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "model_weight.pth"

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.agent = RainbowDQNAgent((4, 84, 84), 12, DEVICE)
        self.agent.load_model(SAVE_PATH)
        self.processor = MarioPreprocessor()
    
    def act(self, observation):
        processed_state = self.processor.process(observation)
        action = self.agent.select_action(processed_state, use_epsilon=False)
        return action