# import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter

# from einops import repeat
from tinygrad import Tensor
from contextlib import contextmanager
import time
import yacs
from yacs.config import CfgNode as CN


def clip_grad_norm_(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return Tensor(0.0)

    def l2_norm(x):
        return Tensor.sqrt(Tensor.sum(Tensor.square(x)))

    norms = [l2_norm(g) for g in grads]
    total_norm = l2_norm(Tensor.stack(norms))
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = Tensor.maximum(clip_coef, 1.0)
    for g in grads:
        g *= clip_coef
        g.realize()
    return total_norm


def cross_entropy(
    x: Tensor, y: Tensor, reduction: str = "mean", label_smoothing: float = 0.0
) -> Tensor:
    divisor = y.shape[1]
    assert isinstance(divisor, int), "only supported int divisor"
    y = (1 - label_smoothing) * y + label_smoothing / divisor
    ret = -x.log_softmax(axis=1).mul(y).sum(axis=1)
    if reduction == "none":
        return ret
    if reduction == "sum":
        return ret.sum()
    if reduction == "mean":
        return ret.mean()


# def clip_grad_norm(parameters:[Tensor], max_norm, norm_type=2):
#     total_norm = 0
#     for p in parameters:
#         param_norm = p.grad.data.norm(norm_type)
#         total_norm += param_norm.item() ** norm_type
#     total_norm = total_norm ** (1. / norm_type)
#     clip_coef = max_norm / (total_norm + 1e-6)
#     if clip_coef < 1:
#         for p in parameters:
#             p.grad.data.mul_(clip_coef)
#     return total_norm
def _sum_rightmost(value, dim):
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def numel(shape):
    return int(np.prod(shape)) if shape else 1


# def seed_np_torch(seed=20001118):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def seed_np(seed=20001118):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class Logger:
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar:
    def __init__(self, decay: float) -> None:
        self.scalar = Tensor(0.0)
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar *= self.decay
        self.scalar += value * (1 - self.decay)
        self.scalar.realize()

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.UseDemonstration = False

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf