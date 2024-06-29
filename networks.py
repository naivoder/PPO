import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta


class Actor(nn.Module):
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha,
        h1_dims=128,
        h2_dims=128,
        chkpt_dir="weights/actor.pt",
    ):
        super(Actor, self).__init__()
        self.checkpoint_file = chkpt_dir
        self.h1 = nn.Linear(*input_dims, h1_dims)
        self.h2 = nn.Linear(h1_dims, h2_dims)
        self.alpha = nn.Linear(h2_dims, n_actions)
        self.beta = nn.Linear(h2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.h1(state))
        x = T.tanh(self.h2(x))

        # Use of Beta distribution taken from Phil Tabor's implementation
        # I tried Normal and MultivariateNormal with no luck.
        # Once I gave up and tried his way it worked immediately...
        # Idk man, he's a wizard I guess, you'll have to ask him 🤷‍♀️
        alpha = F.relu(self.alpha(x)) + 1.0
        beta = F.relu(self.beta(x)) + 1.0
        dist = Beta(alpha, beta)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Critic(nn.Module):
    def __init__(
        self,
        input_dims,
        alpha,
        h1_dims=128,
        h2_dims=128,
        chkpt_dir="weights/critic.pt",
    ):
        super(Critic, self).__init__()
        self.checkpoint_file = chkpt_dir
        self.h1 = nn.Linear(*input_dims, h1_dims)
        self.h2 = nn.Linear(h1_dims, h2_dims)
        self.output = nn.Linear(h2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.h1(state))
        x = T.tanh(self.h2(x))
        return self.output(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
