import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """ Actor (Policy) model implementing dueling networks:

        s -> [Q(s, a) for a in Actions]

    The network structure for the dueling networks implements distinct layers to compute the
    value function and the advantage function:

    state -> Linear(state_size, fc1_units) -> ReLU -> x

        x -> Linear(fc1_units, fc2_units / 2) -> ReLU -> Linear(fc2_units / 2, 1) -> V(s)
        x -> Linear(fc1_units, fc2_units / 2) -> ReLU -> Linear(fc2_units / 2, action_size) -> A(s, a)

    The estimates for value and advantage are combined as follows:

        Q(s, a) = V(s) + (A(s, a) - mean[A(s, a_i) for action a_i])

    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """ Initialize parameters and build model

        :param state_size:  (int) Dimension of the state space
        :param action_size:  (int) Dimension of the action space
        :param seed:  (int) Random seed (int)
        :param units: (tuple)  Number of units in each hidden layer
        :param fc1_units:  (int) Number of nodes on first hidden layer
        :param fc2_units:  (int) Number of nodes on second hidden layer, split between V and A networks
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Shared hidden layer
        self.fc1 = nn.Linear(state_size, fc1_units)

        # Split second hidden layer nodes between value and advantage networks
        n_value_nodes = fc2_units // 2
        n_advantage_nodes = fc2_units - n_value_nodes

        # Value network
        self.fc2_value = nn.Linear(fc1_units, n_value_nodes)
        self.value = nn.Linear(n_value_nodes, 1)
        
        # Advantage network
        self.fc2_advantage = nn.Linear(fc1_units, n_advantage_nodes)
        self.advantage = nn.Linear(n_advantage_nodes, action_size)

    def forward(self, state):
        """ Evaluate the network for a given state. """

        # Shared network layer
        x = F.relu(self.fc1(state))

        # Value network
        x_value = F.relu(self.fc2_value(x))
        x_value = self.value(x_value)
                
        # Advantage network
        x_advantage = F.relu(self.fc2_advantage(x))
        x_advantage = self.advantage(x_advantage)
        
        # Combining value and advantage into Q estimate
        # Q(a, s) = V(s) + (A(s, a) - 1/n sum_i A(s, a_i))
        output = x_value + (x_advantage - torch.mean(x_advantage, 1, keepdim=True))
        return output
