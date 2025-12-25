"""
Deep Q-Network (DQN) Agent for Traffic Signal Control
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from collections import deque, namedtuple

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    """Experience replay buffer to store and sample past experiences"""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=batch_size)
        
        # Convert to tensor batches
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of memory"""
        return len(self.memory)

class DQNetwork(nn.Module):
    """Neural network for Deep Q-Learning"""
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        """Initialize parameters and build model"""
        super(DQNetwork, self).__init__()
        
        # Add input layer
        layers = [nn.Linear(state_size, hidden_layers[0]), nn.ReLU()]
        
        # Add hidden layers
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
        
        # Add output layer
        layers.append(nn.Linear(hidden_layers[-1], action_size))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass through the network"""
        return self.model(state)

class DQNAgent:
    """Agent implementing Deep Q-Learning for traffic signal control"""
    
    def __init__(self, config_file="config.yaml"):
        """Initialize the DQN Agent"""
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        rl_config = config['reinforcement_learning']
        
        # Get parameters from config
        self.learning_rate = rl_config['learning_rate']
        self.gamma = rl_config['discount_factor']
        self.epsilon = rl_config['exploration_rate']
        self.batch_size = rl_config['batch_size']
        self.hidden_layers = rl_config['hidden_layers']
        self.target_update = rl_config['target_update_frequency']
        
        # State and action dimensions (will be set when initializing with environment)
        self.state_size = None
        self.action_size = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks, optimizer and memory
        self.qnetwork_local = None
        self.qnetwork_target = None
        self.optimizer = None
        self.memory = ReplayMemory(rl_config['memory_size'])
        
        # Training parameters
        self.t_step = 0
    
    def initialize(self, state_size, action_size):
        """Initialize networks and optimizer with environment dimensions"""
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize Q-Networks (local and target)
        self.qnetwork_local = DQNetwork(state_size, action_size, self.hidden_layers).to(self.device)
        self.qnetwork_target = DQNetwork(state_size, action_size, self.hidden_layers).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
    
    def step(self, state, action, reward, next_state, done):
        """Add experience to memory and learn if enough samples"""
        # Reshape state if needed
        if isinstance(state, np.ndarray) and len(state.shape) > 1:
            state = state.reshape(1, -1)
        
        if isinstance(next_state, np.ndarray) and len(next_state.shape) > 1:
            next_state = next_state.reshape(1, -1)
            
        # Add experience to memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every few steps
        self.t_step = (self.t_step + 1) % self.target_update
        if self.t_step == 0:
            # Learn if enough samples in memory
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self._learn(experiences)
    
    def act(self, state, eps=None):
        """Return action based on current policy"""
        if eps is None:
            eps = self.epsilon
            
        # Convert state to tensor
        if isinstance(state, np.ndarray) and len(state.shape) > 1:
            state = state.reshape(1, -1)
            
        state = torch.from_numpy(state).float().to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            # Get action values from network
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            
            # Choose best action
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Random action
            return random.choice(np.arange(self.action_size))
    
    def _learn(self, experiences):
        """Update value parameters using batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, 0.01)
    
    def _soft_update(self, local_model, target_model, tau):
        """Soft update target network parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, path):
        """Save model weights"""
        torch.save(self.qnetwork_local.state_dict(), path)
    
    def load(self, path):
        """Load model weights"""
        self.qnetwork_local.load_state_dict(torch.load(path))
        self.qnetwork_target.load_state_dict(torch.load(path))

if __name__ == "__main__":
    # Test DQN implementation
    agent = DQNAgent()
    
    # Sample state and action space sizes
    state_size = 24  # 8 lanes x 3 features
    action_size = 4  # 4 traffic light phases
    
    # Initialize agent
    agent.initialize(state_size, action_size)
    
    # Test action selection
    state = np.random.rand(1, state_size)
    action = agent.act(state)
    print(f"Selected action: {action}")
