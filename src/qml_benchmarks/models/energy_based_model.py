# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flax.linen as nn
from qml_benchmarks.models.base import EnergyBasedModel, BaseGenerator
from sklearn.neural_network import BernoulliRBM
from qml_benchmarks.model_utils import mmd_loss, median_heuristic
import numpy as np
import jax
import jax.numpy as jnp
import itertools

class MLP(nn.Module):
    "Multilayer perceptron implemented in flax"
    # Create a MLP with hidden layers and neurons specfied as a list of integers.
    hidden_layers: list[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_layers:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


class DeepEBM(EnergyBasedModel):
    """
    Energy-based model which uses a fully connected multi-layer perceptron neural network as its energy function.
    The model is trained via k-contrastive divergence.
    The score function corresponds to the (negative of the) maximum mean discrepancy distance.

    Args:
        learning_rate (float): Initial learning rate for training.
        batch_size (int): Size of batches used for computing parameter updates.
        max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
        cdiv_steps (int): number of mcmc steps to perform to estimate the constrastive divergence loss (default 1)
        convergence_interval (int): The number of loss values to consider to decide convergence.
        jit (bool): Whether to use just in time compilation.
        random_state (int): Seed used for pseudorandom number generation.
        hidden_layers (list[int]):
            The number of hidden layers and neurons in the MLP layers. e.g. [8,4] uses a three layer network where the
            first layers maps to 8 neurons, the second to 4, and the last layer to 1 neuron.
        mmd_kwargs (dict): arguments used for the maximum mean discrepancy score. n_samples and n_steps are the args
            sent to self.sample when sampling configurations used for evaluation. sigma is the bandwidth of the
            maximum mean discrepancy.
    """

    def __init__(
        self,
        learning_rate=0.001,
        batch_size=32,
        max_steps=10000,
        cdiv_steps=1,
        convergence_interval=None,
        random_state=42,
        jit=True,
        hidden_layers=[8, 4],
        mmd_kwargs={"n_samples": 1000, "n_steps": 1000, "sigma": 1.0},
    ):
        super().__init__(
            dim=None,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            cdiv_steps=cdiv_steps,
            convergence_interval=convergence_interval,
            random_state=random_state,
            jit=jit,
        )
        self.hidden_layers = hidden_layers
        self.mmd_kwargs = mmd_kwargs

    def initialize(self, x):
        dim = x.shape[1]
        if not isinstance(dim, int):
            raise NotImplementedError(
                "The model is not yet implemented for data"
                "with arbitrary dimensions. `dim` must be an integer."
            )

        self.dim = dim
        self.model = MLP(hidden_layers=self.hidden_layers)
        self.params_ = self.model.init(self.generate_key(), x)

    def energy(self, params, x):
        """
        energy function
        Args:
            params: parameters of the neural network to be passed to flax
            x: batch of configurations of shape (n_batch, dim)
        returns:
            batch of energy values
        """
        return self.model.apply(params, x)

    def score(self, X: np.ndarray, y=None) -> float:
        """
        Maximum mean discrepancy score function
        Args:
            X (Array): batch of test samples to evalute the model against.
        """
        sigma = self.mmd_kwargs["sigma"]
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        score = np.mean(
            [
                mmd_loss(
                    X,
                    self.sample(
                        self.mmd_kwargs["n_samples"], self.mmd_kwargs["n_steps"]
                    ),
                    sigma,
                )
                for sigma in sigmas
            ]
        )
        return float(-score)


class RestrictedBoltzmannMachine(BernoulliRBM, BaseGenerator):
    """
    Implementation of a restricted Boltzmann machine. The model wraps the scikit-learn BernoulliRBM class and is
    trained via constrastive divergence.
    Args:
        n_components (int): Number of hidden units in the RBM
        learning_rate (float): learning rate for training
        batch_size (int): batch size for training
        n_iter (int): number of epochs of training
        verbose (int): verbosity level
        random_state (int): random seed used for reproducibility
        score_fn (str): determinies the score function used in hyperparameter optimization. If 'pseudolikelihood'
            sklearn's pseudolikelihood function is used, if 'mmd' the (negative of) the maximum mean discrepancy is used.
        mmd_kwargs (dict): arguments used for the maximum mean discrepancy score. n_samples and n_steps are the args
            sent to self.sample when sampling configurations used for evaluation. sigma is the bandwidth of the
            maximum mean discrepancy.

    """

    def __init__(
        self,
        n_components=256,
        learning_rate=0.0001,
        batch_size=10,
        n_iter=10,
        verbose=0,
        random_state=42,
        score_fn="pseudolikelihood",
        mmd_kwargs={"n_samples": 1000, "n_steps": 1000, "sigma": 1.0},
    ):
        super().__init__(
            n_components=n_components,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_iter=n_iter,
            verbose=verbose,
            random_state=random_state,
        )
        self.score_fn = score_fn
        self.mmd_kwargs = mmd_kwargs
        self.rng = np.random.default_rng(random_state)

    def initialize(self, X: any = None):
        if len(X.shape) > 2:
            raise ValueError("Input data must be 2D")
        self.dim = X.shape[1]

    def fit(self, X, y=None):
        """
        fit the model using k-contrastive divergence. simply wraps the sklearn fit function.
        Args:
            X (np.array): training data
            y: not used; set to None to interface with sklearn correctly.
        """
        self.initialize(X)
        super().fit(X, y)

    # Gibbs sampling:
    def _sample(self, init_configs, num_steps=1000):
        """
        Sample the model for given number of steps via the .gibbs method of sklean's RBM. The initial configuration
        is sampled randomly.

        Args:
            num_steps (int): Number of Gibbs sample steps

        Returns:
            np.array: The sampled configurations
        """
        if self.dim is None:
            raise ValueError("Model must be initialized before sampling")
        v = init_configs
        for _ in range(num_steps):
            v = self.gibbs(v)  # Assuming `gibbs` is an instance method
        return v

    def sample(self, num_samples: int, num_steps: int = 1000) -> np.ndarray:
        """
        Sample the model. Each sample is generated by sampling a random configuration and performing a number of
            Gibbs sampling steps. We use joblib to parallelize the sampling.
        Args:
            num_samples (int): number of samples to return
            num_steps (int): number of Gibbs sampling steps for each sample
        """
        init_configs = self.rng.choice([0, 1], size=(num_samples, self.dim,))
        samples_t = self._sample(init_configs, num_steps=num_steps)
        samples_t = np.array(samples_t, dtype=int)
        return samples_t

    def energy(self, v, h):
        """
        The energy for a given visible and hidden configuration
        Args:
            v (np.array): visible configuration
            h (np.array): hidden configuration
        """
        c = jnp.array(self.intercept_hidden_)
        b = jnp.array(self.intercept_visible_)
        W = jnp.array(self.components_)
        return - h @ W @ v - jnp.dot(b, v) - jnp.dot(c, h)

    def compute_partition_function(self):
        """
        computes the partition function. Note this scales exponentially with the total number of neurons and
        is therefore only suitable for small models
        """

        print('computing partition fn...')

        def increment_partition_fn(i, val):
            v = all_bitstrings[i, :self.dim]
            h = all_bitstrings[i, self.dim:]
            return val + jnp.exp(-self.energy(v, h))

        all_bitstrings = jnp.array(list(itertools.product([0, 1], repeat=self.dim + self.n_components)))

        self.partition_function = jax.lax.fori_loop(0, all_bitstrings.shape[0], increment_partition_fn, 0)

        return self.partition_function

    def probability(self, v):
        """
        Compute the probability of a visible configuration. Requires computation of partition function and is
        therefore only suitable for small models.
        Args:
            v (np.array): A visible configuration
        """
        def increment_visible_prob(i, val):
            return val + jnp.exp(-self.energy(v, hidden_bitstrings[i]))

        if not(hasattr(self, 'partition_function')):
            self.compute_partition_function()

        hidden_bitstrings = jnp.array(list(itertools.product([0, 1], repeat=self.n_components)))
        hidden_sum = jax.lax.fori_loop(0, hidden_bitstrings.shape[0], increment_visible_prob, 0)
        return hidden_sum / self.partition_function

    def visible_probabilities(self):
        """
        Compute all visible probabilities. Requires computation of partition function and is
        therefore only suitable for small models.
        """
        @jax.jit
        def prob(v):
            return self.probability(v)

        visible_bitstrings = jnp.array(list(itertools.product([0, 1], repeat=self.dim)))
        return np.array([prob(v) for v in visible_bitstrings])

    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """
        Score function for hyperparameter optimization.
        Args:
            X (Array): batch of test samples to evalute the model against.
        """
        if self.score_fn == "pseudolikelihood":
            return float(np.mean(super().score_samples(X)))
        elif self.score_fn == "mmd":
            sigma = self.mmd_kwargs["sigma"]
            sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
            score = np.mean(
                [
                    mmd_loss(
                        X,
                        self.sample(
                            self.mmd_kwargs["n_samples"], self.mmd_kwargs["n_steps"]
                        ),
                        sigma,
                    )
                    for sigma in sigmas
                ]
            )
            return float(-score)
