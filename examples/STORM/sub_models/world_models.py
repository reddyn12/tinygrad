# import torch
# import torch.nn as tnn
# import torch.nn.functional as F

from tinygrad import Tensor, dtypes, nn, TinyJit
from tinygrad.nn.state import get_state_dict, get_parameters
from distributions import OneHotCategorical, Normal
import distributions

# from einops import rearrange, repeat, reduce
# from einops.layers.torch import Rearrange
# from torch.cuda.amp import autocast
from utils import cross_entropy, clip_grad_norm_
from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import (
    get_subsequent_mask_with_batch_length,
    get_subsequent_mask,
)
from sub_models.transformer_model import StochasticTransformerKVCache
import agents


class EncoderBN:
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        # super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        feature_width = 64 // 2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        # backbone.append(nn.ReLU(inplace=True))
        backbone.append(Tensor.relu)

        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            # backbone.append(nn.ReLU(inplace=True))
            backbone.append(Tensor.relu)

            if feature_width == final_feature_width:
                break

        # self.backbone = nn.Sequential(*backbone)
        # self.backbone = Tensor.sequential(backbone)
        self.backbone = backbone
        self.last_channels = channels

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        # x = rearrange(x, "B L C H W -> (B L) C H W")
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        # x = self.backbone(x)
        x = x.sequential(self.backbone)
        # x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        # x = x.repeat((batch_size,1,1,1,1))
        # x = x.reshape(x.shape[0],x.shape[1],
        #                                      x.shape[2]*x.shape[3]*x.shape[4])
        x = x.reshape(
            batch_size, x.shape[0] // batch_size, x.shape[1] * x.shape[2] * x.shape[3]
        )
        return x

    def __call__(self, x: Tensor):
        return self.forward(x)


class DecoderBN:
    def __init__(
        self,
        stoch_dim,
        last_channels,
        original_in_channels,
        stem_channels,
        final_feature_width,
    ) -> None:
        # super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Linear(
                stoch_dim,
                last_channels * final_feature_width * final_feature_width,
                bias=False,
            )
        )
        # backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_channels, H=final_feature_width))
        backbone.append(
            lambda x: x.reshape(
                x.shape[0] * x.shape[1],
                last_channels,
                final_feature_width,
                x.shape[2] // final_feature_width // last_channels,
            )
        )
        backbone.append(nn.BatchNorm2d(last_channels))
        # backbone.append(nn.ReLU(inplace=True))
        backbone.append(Tensor.relu)
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            # backbone.append(nn.ReLU(inplace=True))
            backbone.append(Tensor.relu)

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.backbone = backbone
        # self.backbone = nn.Sequential(*backbone)
        # self.backbone = Tensor.sequential(backbone)

    def forward(self, sample: Tensor):
        batch_size = sample.shape[0]
        # obs_hat = self.backbone(sample)
        obs_hat = sample.sequential(self.backbone)
        # obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        obs_hat = obs_hat.reshape(
            batch_size,
            obs_hat.shape[0] // batch_size,
            obs_hat.shape[1],
            obs_hat.shape[2],
            obs_hat.shape[3],
        )
        return obs_hat

    def __call__(self, sample):
        return self.forward(sample)


class DistHead:
    """
    Dist: abbreviation of distribution
    """

    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        # super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim * stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim * stoch_dim)

    def unimix(self, logits: Tensor, mixing_ratio=0.01):
        # uniform noise mixing
        # probs = F.softmax(logits, dim=-1)
        probs = logits.softmax(axis=-1)
        # mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        mixed_probs = (
            mixing_ratio * Tensor.ones_like(probs) / self.stoch_dim
            + (1 - mixing_ratio) * probs
        )
        # logits = torch.log(mixed_probs)
        logits = mixed_probs.log()
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        # logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = logits.reshape(
            logits.shape[0],
            logits.shape[1],
            self.stoch_dim,
            logits.shape[2] // self.stoch_dim,
        )
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        # logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = logits.reshape(
            logits.shape[0],
            logits.shape[1],
            self.stoch_dim,
            logits.shape[2] // self.stoch_dim,
        )
        logits = self.unimix(logits)
        return logits


class RewardDecoder:
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
        # super().__init__()
        self.backbone = [
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            # nn.ReLU(inplace=True),
            Tensor.relu,
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            # nn.ReLU(inplace=True),
            Tensor.relu,
        ]
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat: Tensor):
        feat = feat.sequential(self.backbone)
        reward = self.head(feat)
        return reward

    def __call__(self, feat):
        return self.forward(feat)


class TerminationDecoder:
    def __init__(self, embedding_size, transformer_hidden_dim) -> None:
        # super().__init__()
        self.backbone = [
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            # nn.ReLU(inplace=True),
            Tensor.relu,
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            # nn.ReLU(inplace=True),
            Tensor.relu,
        ]
        self.head = [
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        ]

    def forward(self, feat: Tensor):
        feat = feat.sequential(self.backbone)
        termination = feat.sequential(self.head)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination

    def __call__(self, feat):
        return self.forward(feat)


class MSELoss:
    def forward(self, obs_hat: Tensor, obs: Tensor):
        loss = (obs_hat - obs) ** 2
        # loss = reduce(loss, "B L C H W -> B L", "sum")
        loss = loss.sum(-1).sum(-1).sum(-1)
        return loss.mean()

    def __call__(self, obs_hat, obs):
        return self.forward(obs_hat, obs)


class CategoricalKLDivLossWithFreeBits:
    def __init__(self, free_bits) -> None:
        # super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        # kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = distributions.kl_divergence(p_dist, q_dist)
        # kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.sum(axis=-1)
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        # kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        kl_div = Tensor.maximum(kl_div, 1 * self.free_bits)
        return kl_div, real_kl_div

    def __call__(self, p_logits, q_logits):
        return self.forward(p_logits, q_logits)


class WorldModel:
    def __init__(
        self,
        in_channels,
        action_dim,
        transformer_max_length,
        transformer_hidden_dim,
        transformer_num_layers,
        transformer_num_heads,
    ):
        self.transformer_hidden_dim = transformer_hidden_dim
        self.final_feature_width = 4
        self.stoch_dim = 32
        self.stoch_flattened_dim = self.stoch_dim * self.stoch_dim
        self.use_amp = False
        # self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.tensor_dtype = dtypes.bfloat16 if self.use_amp else dtypes.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1

        self.encoder = EncoderBN(
            in_channels=in_channels,
            stem_channels=32,
            final_feature_width=self.final_feature_width,
        )
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1,
        )
        self.dist_head = DistHead(
            image_feat_dim=self.encoder.last_channels
            * self.final_feature_width
            * self.final_feature_width,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=self.stoch_dim,
        )
        self.image_decoder = DecoderBN(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=in_channels,
            stem_channels=32,
            final_feature_width=self.final_feature_width,
        )
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim,
        )
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim,
        )

        self.mse_loss_func = MSELoss()
        # self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss = lambda x, y: cross_entropy(x, y)
        # self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.bce_with_logits_loss_func = Tensor.binary_crossentropy_logits
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(
            num_classes=255, lower_bound=-20, upper_bound=20
        )
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.optimizer = nn.optim.Adam(self.parameters(), lr=1e-4)
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def parameters(self):
        models = [
            self.encoder,
            self.storm_transformer,
            self.dist_head,
            self.image_decoder,
            self.reward_decoder,
            self.termination_decoder,
        ]
        return get_parameters(models)

    def encode_obs(self, obs):
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
        embedding = self.encoder(obs)
        post_logits = self.dist_head.forward_post(embedding)
        sample = self.stright_throught_gradient(
            post_logits, sample_mode="random_sample"
        )
        flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
        temporal_mask = get_subsequent_mask(latent)
        dist_feat = self.storm_transformer(latent, action, temporal_mask)
        last_dist_feat = dist_feat[:, -1:]
        prior_logits = self.dist_head.forward_prior(last_dist_feat)
        prior_sample = self.stright_throught_gradient(
            prior_logits, sample_mode="random_sample"
        )
        prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, last_flattened_sample, action, log_video=True):
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
        dist_feat = self.storm_transformer.forward_with_kv_cache(
            last_flattened_sample, action
        )
        prior_logits = self.dist_head.forward_prior(dist_feat)

        # decoding
        prior_sample = self.stright_throught_gradient(
            prior_logits, sample_mode="random_sample"
        )
        prior_flattened_sample = self.flatten_sample(prior_sample)
        if log_video:
            obs_hat = self.image_decoder(prior_flattened_sample)
        else:
            obs_hat = None
        reward_hat = self.reward_decoder(dist_feat)
        reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
        termination_hat = self.termination_decoder(dist_feat)
        termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample: Tensor):
        # return rearrange(sample, "B L K C -> B L (K C)")
        return sample.reshape(
            sample.shape[0], sample.shape[1], sample.shape[2] * sample.shape[3]
        )

    def imagine_data(
        self,
        agent: agents.ActorCriticAgent,
        sample_obs,
        sample_action,
        imagine_batch_size,
        imagine_batch_length,
        log_video,
        logger,
    ):
        obs_hat_list = []
        latent_buffer = []
        hidden_buffer = []
        action_buffer = []
        reward_hat_list = []
        termination_hat_list = []

        self.storm_transformer.reset_kv_cache_list(
            imagine_batch_size, dtype=self.tensor_dtype
        )
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            (
                last_obs_hat,
                last_reward_hat,
                last_termination_hat,
                last_latent,
                last_dist_feat,
            ) = self.predict_next(
                context_latent[:, i : i + 1],
                sample_action[:, i : i + 1],
                log_video=log_video,
            )
        latent_buffer.append(last_latent)
        hidden_buffer.append(last_dist_feat)

        # imagine
        for i in range(imagine_batch_length):
            # action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            action = agent.sample(Tensor.cat(last_latent, last_dist_feat, dim=-1))
            action_buffer.append(action)

            (
                last_obs_hat,
                last_reward_hat,
                last_termination_hat,
                last_latent,
                last_dist_feat,
            ) = self.predict_next(last_latent, action, log_video=log_video)

            latent_buffer.append(last_latent)
            hidden_buffer.append(last_dist_feat)
            reward_hat_list.append(last_reward_hat)
            termination_hat_list.append(last_termination_hat)
            if log_video:
                obs_hat_list.append(
                    last_obs_hat[:: imagine_batch_size // 16]
                )  # uniform sample vec_env

        if log_video:
            # logger.log("Imagine/predict_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())
            logger.log(
                "Imagine/predict_video",
                Tensor.clip(Tensor.cat(obs_hat_list, dim=1), 0.0, 1.0).float().numpy(),
            )

        # Convert buffers to tensors
        latent_buffer = Tensor.stack(latent_buffer, dim=1)
        hidden_buffer = Tensor.stack(hidden_buffer, dim=1)
        action_buffer = Tensor.stack(action_buffer, dim=1)
        reward_hat_buffer = Tensor.stack(reward_hat_list, dim=1)
        termination_hat_buffer = Tensor.stack(termination_hat_list, dim=1)

        # return torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer
        return (
            Tensor.cat(*[latent_buffer, hidden_buffer], dim=-1),
            action_buffer,
            reward_hat_buffer,
            termination_hat_buffer,
        )

    @TinyJit
    def update(self, obs, action, reward, termination):
        # self.train()
        batch_size, batch_length = obs.shape[:2]

        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
        # encoding
        embedding = self.encoder(obs)
        post_logits = self.dist_head.forward_post(embedding)
        sample = self.stright_throught_gradient(
            post_logits, sample_mode="random_sample"
        )
        flattened_sample = self.flatten_sample(sample)

        # decoding image
        obs_hat = self.image_decoder(flattened_sample)

        # transformer
        temporal_mask = get_subsequent_mask_with_batch_length(
            batch_length, flattened_sample.device
        )  # .realize()
        dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
        prior_logits = self.dist_head.forward_prior(dist_feat)
        # decoding reward and termination with dist_feat
        reward_hat = self.reward_decoder(dist_feat)
        termination_hat = self.termination_decoder(dist_feat)

        # env loss
        reconstruction_loss = self.mse_loss_func(obs_hat, obs)
        reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
        termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
        # dyn-rep loss
        dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(
            post_logits[:, 1:].detach(), prior_logits[:, :-1]
        )
        representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(
            post_logits[:, 1:], prior_logits[:, :-1].detach()
        )
        total_loss = (
            reconstruction_loss
            + reward_loss
            + termination_loss
            + 0.5 * dynamics_loss
            + 0.1 * representation_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        max_norm = 1000.0
        clip_grad_norm_(self.parameters(), max_norm)
        self.optimizer.step()

        metrics = {
            "WorldModel/reconstruction_loss": reconstruction_loss.realize(),
            "WorldModel/reward_loss": reward_loss.realize(),
            "WorldModel/termination_loss": termination_loss.realize(),
            "WorldModel/dynamics_loss": dynamics_loss.realize(),
            "WorldModel/dynamics_real_kl_div": dynamics_real_kl_div.realize(),
            "WorldModel/representation_loss": representation_loss.realize(),
            "WorldModel/representation_real_kl_div": representation_real_kl_div.realize(),
            "WorldModel/total_loss": total_loss.realize(),
        }

        return metrics