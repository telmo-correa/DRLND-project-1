import numpy as np


class SumTree:
    """ Helper data structure for Prioritized Experience Replay memory buffer.

    Probability weights for indexes are stored in a circular array.  Indexes
    may be sampled randomly with replacement according to the probability
    weights by using a helper binary tree, constructed at time of drawing.

    The elements themselves are not stored in this data structure, and should
    be saved on a circular array / deque with same capacity for indexing.


    Sampling weights

    The stored probability weights are pre-processed in the following manner,
    as in the PER paper:

        P[i] := (p[i] ** alpha) / sum[ p[j] ** alpha ]
        w[i] := (1 / (N * P[i]) ** beta

    The weight used for sampling is w[i] * p[i].


    Sampling

    The binary tree is represented as an array of shape (num_layers, capacity),
    with each row having elements that are the sum of the two corresponding
    child elements:

        layers[i + 1, k] := layers[i, 2 * k] + layers[i, 2 * k + 1]

    Layers are computed in a single pass of vector operations.  Sampling is
    done in a batch, also with a pass of vector operations.


    Usage

    * Initialize the data structure with a fixed capacity
    * To add new elements, call tree.add(n_elements).  New elements are added
    with the default initial weight equal to the maximum weight currently
    stored, or 1 if there are no elements stored.
    * To sample elements:
        * First, call tree.sample(batch_size, alpha, beta) to obtain a list of
        indexes sampled according to the specified probability distribution
        * Then, call tree.update(indexes, weights) to update the probability
        weights associated with the sampled indexes
    """

    def __init__(self, capacity, seed=0):
        """ Initializes the data structure with a fixed max capacity.

        :param capacity:  (int) Maximum number of elements to store
        :param seed:  (int) Seed used for PRNG
        """
        self.capacity = capacity
        self.errors = np.zeros(2 * SumTree._half_round_up(capacity))
        self.num_layers = int(np.ceil(np.log2(capacity))) + 1
        self.layers = np.zeros((self.num_layers, 2 * SumTree._half_round_up(capacity)))
        self.current_size = 0
        self.cursor = 0

        np.random.seed(seed)

    def add(self, n_elements):
        """ Adds N indexes with weight p_max (or 1, if there are no elements)

        :param n_elements:  (int) Number of elements with weight 0 to add
        """
        # If no elements being added, return
        if n_elements == 0:
            return

        # When adding more elements than capacity, only use the ones that will fit
        if n_elements > self.capacity:
            self.add(self.capacity)
            return

        # Added indexes loop around array, divide in two calls
        if (self.cursor + n_elements) > self.capacity:
            stop_index = self.capacity - self.cursor
            self.add(stop_index)
            self.add(n_elements - stop_index)
            return

        self.errors[self.cursor:(self.cursor + n_elements)] = 1 if self.current_size == 0 else self.errors.max()

        # Update cursors
        self.cursor += n_elements
        if self.cursor >= self.capacity:
            self.cursor -= self.capacity
        self.current_size = np.minimum(self.current_size + n_elements, self.capacity)

    def sample(self, batch_size=1, alpha=1, beta=1):
        """ Samples a given number of elements.

        :param batch_size:  (int) Number of elements to sample
        :param alpha: (float) Prioritization parameter
        :param beta: (float) Bias-annealling parameter
        :return:  batch_index:  (array indexes) Index of sampled elements
        """
        # Compute values to propagate through data structure
        errors_exp_alpha = self.errors ** alpha
        P_i = errors_exp_alpha / np.sum(errors_exp_alpha)
        with np.errstate(divide='ignore'):  # Suppress warnings for division by zero on masked array
            weights = np.where(self.errors == 0, 0, (1 / (self.current_size * P_i)) ** beta)
        self.layers[0] = weights * self.errors

        # Propagate values through tree structure
        max_index = self.layers.shape[1]
        for i in range(1, self.num_layers):
            n = SumTree._half_round_up(max_index)
            self.layers[i, :n] = self.layers[i - 1, :2*n:2] + self.layers[i - 1, 1:2*n:2]
            max_index = n

        # Sample from tree using a starting uniform distribution
        batch_seeds = np.random.uniform(low=0, high=self.layers[-1, 0], size=batch_size)
        batch_index = np.zeros(batch_size, dtype=np.uint32)

        for layer_index in range(-2, -self.num_layers - 1, -1):
            right_nodes = (batch_seeds > self.layers[layer_index, batch_index])
            batch_seeds -= self.layers[layer_index, batch_index] * right_nodes
            batch_index = 2 * batch_index + right_nodes

        batch_index = np.minimum(batch_index, self.current_size - 1)
        return batch_index

    def update(self, indexes, weights):
        """ Updates the weights for the given indexes

        :param indexes:  (array indexes) Indexes to update
        :param weights:  (array values) New weight values
        """
        self.layers[0, indexes] = weights

    @staticmethod
    def _half_round_up(x):
        return x // 2 if x % 2 == 0 else x // 2 + 1

    def __len__(self):
        """ Return the number of elements added to the data structure, up to its capacity. """
        return self.current_size
