import torch
import numpy as np


class Actor(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        min_action,
        max_action,
        learning_rate=3e-4,
        h1_size=256,
        h2_size=256,
        reparam_noise=1e-6,
        chkpt_dir="weights/actor.pt",
    ):
        super(Actor, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = learning_rate
        self.min_action = min_action
        self.max_action = max_action
        self.reparam_noise = reparam_noise
        self.checkpoint_path = chkpt_dir

        self.fc1 = torch.nn.Linear(*input_shape, self.h1_size)
        self.fc2 = torch.nn.Linear(self.h1_size, self.h2_size)
        self.mean = torch.nn.Linear(self.h2_size, n_actions)
        self.std = torch.nn.Linear(self.h2_size, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.action_scale = torch.FloatTensor(
            (self.max_action - self.min_action) / 2.0
        ).to(self.device)
        self.action_bias = torch.FloatTensor(
            (self.max_action + self.min_action) / 2.0
        ).to(self.device)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)
        std = torch.clamp(std, -20, 2)
        return mean, std

    def sample_normal(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        probs = torch.distributions.Normal(mu, std)

        x_t = probs.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_probs = probs.log_prob(x_t)
        log_probs -= torch.log(
            self.action_scale * (1 - y_t.pow(2)) + self.reparam_noise
        )
        log_probs = log_probs.sum(1, keepdim=True)

        # for deterministic policy return mu instead of action
        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))


class Critic(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        alpha=3e-4,
        h1_size=256,
        h2_size=256,
        chkpt_dir="weights/critic.pt",
    ):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.alpha = alpha
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.chkpt_dir = chkpt_dir

        self.h1_layer = torch.nn.Linear(np.prod(self.input_dims), self.h1_size)
        self.h2_layer = torch.nn.Linear(self.h1_size, self.h2_size)
        self.output = torch.nn.Linear(self.h2_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), self.alpha, amsgrad=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.h1_layer(x))
        x = torch.nn.functional.tanh(self.h2_layer(x))
        return self.output(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_dir))
