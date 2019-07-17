import numpy as np
import random

from model import DuelingQNetwork
from memory import PrioritizedReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """ Interacts with and learns from the environment.

    This agent implements a few improvements over the vanilla DQN, making it
    a Double Dueling Deep Q-Learning Network with Prioritized Experience Replay.

    * Deep Q-Learning Network:  RL where a deep learning network
      is used for the Q-network estimate.
    * Double DQN:  The local network from DQN is used to select the
      optimal action during learning, but the policy estimate for
      that action is computed using the target network.
    * Dueling DQN:  The deep learning network explicitly estimates
      the value function and the advantage functions separately.
    * DQN-PER:  Experiences are associated with a probability weight
      based upon the absolute error between the estimated Q-value
      and the target Q-value at time of estimation -- prioritizing
      experiences that help learn more.
    """

    def __init__(
        self, 
        state_size, 
        action_size, 
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        learn_rate=5e-4,
        update_every=4,
        per_epsilon=1e-5,
        per_alpha=0.6,
        per_beta=0.9,
        device=DEFAULT_DEVICE,
        seed=0
    ):
        """ Initialize an object.

        :param state_size:  (int) Dimension of each state
        :param action_size:  (int) Dimension of each action
        :param buffer_size:  (int) Replay buffer size
        :param batch_size:  (int) Minibatch size used during learning
        :param gamma:  (float) Discount factor
        :param tau:  (float) Scaling parameter for soft update
        :param learn_rate:  (float) Learning rate used by optimizer
        :param update_every:  (int) Steps between updates of target network
        :param per_epsilon:  (float) PER hyperparameter, constant added to each error
        :param per_alpha:  (float) PER hyperparameter, exponent applied to each probability
        :param per_beta:  (float) PER hyperparameter, bias correction exponent for probability weight
        :param device:  (torch.device)  Object representing the device where to allocate tensors
        :param seed:  (int) Seed used for PRNG
        """
        # Save copy of model parameters
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device

        # Save copy of hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learn_rate = learn_rate
        self.update_every = update_every
        self.per_epsilon = per_epsilon
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        
        # Q networks
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learn_rate)
        
        # Replay memory
        self.memory = PrioritizedReplayBuffer(
            memory_size=buffer_size,
            device=device,
            update_every=update_every,
            seed=seed
        )
        
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0
        self.episode = 0
    
    def step(self, state, action, reward, next_state, done):
        """ Store a single agent step, learning every N steps

        :param state: (array-like) Initial state on the visit
        :param action: (int) Action on the visit
        :param reward: (float) Reward received on the visit
        :param next_state:  (array-like) State reached after the visit
        :param done:  (bool) Flag whether the next state is a terminal state
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every self.update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(batch_size=self.batch_size, alpha=self.per_alpha, beta=self.per_beta)
                self.learn(experiences)

        # Keep track of episode number
        if done:
            self.episode += 1

    def act(self, state, eps=0.):
        """ Returns the selected action for the given state according to the current policy

        :param state: (array_like) Current state
        :param eps: (float) Epsilon, for epsilon-greedy action selection
        :return: action (int)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        # Convert types to np.int32 for compatibility with environment 
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int32)

    def learn(self, experiences):
        """ Update value parameters using given batch of indexed experience tuples

        :param experiences:  (Tuple[torch.Tensor, np.array]) (s, a, r, s', done, index) tuples
        """
        states, actions, rewards, next_states, dones, indexes = experiences

        # Get max predicted Q values (for next states) from target model

        # Double DQN: use local network to select action with maximum value, 
        # then use target network to get Q value for that action        
        Q_next_indices = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_next_values = self.qnetwork_target(next_states).detach()
        Q_targets_next = Q_next_values.gather(1, Q_next_indices)

        # Compute Q target for current states 
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute estimation error (for Prioritized Experience Replay) and update weights
        Q_error = (torch.abs(Q_expected.detach() - Q_targets.detach()) + self.per_epsilon).squeeze()
        self.memory.update(indexes, Q_error)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
