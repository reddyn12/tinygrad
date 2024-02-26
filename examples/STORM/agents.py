# import torch
# import torch.nn as tnn
# import torch.nn.functional as F
from tinygrad import Tensor, dtypes, nn, TinyJit
from tinygrad.nn.state import get_state_dict, get_parameters

import tinygrad

# import torch.distributions as distributions
import distributions

# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
import copy

# from torch.cuda.amp import autocast

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar, clip_grad_norm_


def sort(arr: Tensor):
    assert arr.ndim == 1, "Cannot sort multidimensional array"
    # O(n) insertion sort, but works with JIT; find a better way
    temp = []
    arrmax = arr.max()
    for _ in range(arr.numel()):
        idx = arr.argmin().realize()
        temp.append(arr[idx])
        arr = Tensor.where(arr == arr[idx], arrmax, arr).realize()
    return Tensor.stack(temp)


def percentile(x: Tensor, percentage: float):
    # flat_x = torch.flatten(x)
    flat_x = x.flatten()
    kth = int(percentage * flat_x.shape[0])  # maybe can do flat_x[kth-1]
    # per = torch.kthvalue(flat_x, kth).values
    sorted_x = sort(flat_x)
    per = sorted_x[kth]
    # import sys
    # print('percentile',per, per.dtype)
    # sys.exit()
    return per


# def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=dtypes.float32):
    # detach all inputs
    rewards = rewards.detach()
    values = values.detach()
    termination = termination.detach()

    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    # gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device="cuda")
    gamma_return = Tensor.zeros((batch_size, batch_length + 1, 1), dtype=dtype)
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = (
            rewards[:, t]
            + gamma * inv_termination[:, t] * (1 - lam) * values[:, t]
            + gamma * inv_termination[:, t] * lam * gamma_return[:, t + 1]
        )
    return gamma_return[:, :-1]


class ActorCriticAgent:
    def __init__(
        self,
        feat_dim,
        num_layers,
        hidden_dim,
        action_dim,
        gamma,
        lambd,
        entropy_coef: float,
    ) -> None:
        # super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.use_amp = False
        # self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.tensor_dtype = dtypes.bfloat16 if self.use_amp else dtypes.float32

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        actor = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            # nn.ReLU()
            Tensor.relu,
        ]
        for i in range(num_layers - 1):
            actor.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    # nn.ReLU()
                    Tensor.relu,
                ]
            )
        actor.append(nn.Linear(hidden_dim, int(action_dim)))

        # self.actor = Tensor.sequential(
        #     actor,
        #     # nn.Linear(hidden_dim, action_dim)
        # )
        self.actor = actor
        critic = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            # nn.ReLU()
            Tensor.relu,
        ]
        for i in range(num_layers - 1):
            critic.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    # nn.ReLU()
                    Tensor.relu,
                ]
            )
        critic.append(nn.Linear(hidden_dim, 255))
        # self.critic = Tensor.sequential([
        #     critic,
        #     # nn.Linear(hidden_dim, 255)
        # ])
        self.critic = critic
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        self.optimizer = nn.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def parameters(self):
        models = [self.actor, self.critic]
        return get_parameters(models)

    # @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        # for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
        for slow_param, param in zip(
            get_parameters(self.slow_critic), get_parameters(self.critic)
        ):
            slow_param *= decay
            slow_param += param.detach() * (1 - decay)
            slow_param.realize()

    def policy(self, x):
        # logits = self.actor(x)
        logits = x.sequential(self.actor)
        return logits

    def value(self, x):
        # value = self.critic(x)
        value = x.sequential(self.critic)
        value = self.symlog_twohot_loss.decode(value)
        return value

    # @torch.no_grad()
    def slow_value(self, x):
        # value = self.slow_critic(x)
        value = x.sequential(self.slow_critic)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        # logits = self.actor(x)
        logits = x.sequential(self.actor)
        raw_value = x.sequential(self.critic)
        return logits, raw_value

    # @torch.no_grad()
    def sample(self, latent, greedy=False):
        # self.eval()
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
        logits = self.policy(latent)
        dist = distributions.Categorical(logits=logits)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        # return action.detach().cpu().squeeze(-1).numpy()
        return action.detach().numpy()

    @TinyJit
    def update(self, latent, action, reward, termination):
        """
        Update policy and value model
        """
        # self.train()
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):

        logits, raw_value = self.get_logits_raw_value(latent)
        dist = distributions.Categorical(logits=logits[:, :-1])
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # decode value, calc lambda return
        slow_value = self.slow_value(latent)
        slow_lambda_return = calc_lambda_return(
            reward, slow_value, termination, self.gamma, self.lambd
        )
        value = self.symlog_twohot_loss.decode(raw_value)
        lambda_return = calc_lambda_return(
            reward, value, termination, self.gamma, self.lambd
        )

        # update value function with slow critic regularization
        value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
        slow_value_regularization_loss = self.symlog_twohot_loss(
            raw_value[:, :-1], slow_lambda_return.detach()
        )

        lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
        upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
        S = upper_bound - lower_bound
        # norm_ratio = torch.max(torch.ones(1).cuda(), S)  # max(1, S) in the paper
        norm_ratio = Tensor.max(Tensor.ones(1), S)  # max(1, S) in the paper
        norm_advantage = (lambda_return - value[:, :-1]) / norm_ratio
        policy_loss = -(log_prob * norm_advantage.detach()).mean()

        entropy_loss = entropy.mean()

        loss = (
            policy_loss
            + value_loss
            + slow_value_regularization_loss
            - self.entropy_coef * entropy_loss
        )

        # # gradient descent
        # self.scaler.scale(loss).backward()
        # self.scaler.unscale_(self.optimizer)  # for clip grad
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        # self.optimizer.zero_grad(set_to_none=True)

        # NEW gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        # Add clip gradient norm
        clip_grad_norm_(self.parameters(), 100.0)

        self.optimizer.step()

        self.update_slow_critic()

        metrics = {
            "ActorCritic/policy_loss": policy_loss.realize(),
            "ActorCritic/value_loss": value_loss.realize(),
            "ActorCritic/entropy_loss": entropy_loss.realize(),
            "ActorCritic/S": S.realize(),
            "ActorCritic/norm_ratio": norm_ratio.realize(),
            "ActorCritic/total_loss": loss.realize(),
        }

        return metrics