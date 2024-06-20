import functools
import os, sys
import time
import os, time, math, functools
from pathlib import Path
from tqdm import tqdm
import multiprocessing

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import colored, getenv, BEAM, WINO, round_up, diskcache_clear
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save
from tinygrad.nn.optim import LAMB, LARS, SGD, OptimizerGroup

from extra.lr_scheduler import LRSchedulerGroup
from examples.mlperf.helpers import get_training_state, load_training_state

def train_resnet():
  from extra.models import resnet
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.datasets.imagenet import get_train_files, get_val_files
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup
  from examples.mlperf.initializers import Conv2dHeNormal, Linear
  from examples.hlb_cifar10 import UnsyncedBatchNorm

  config = {}
  seed = config["seed"] = getenv("SEED", 42)
  Tensor.manual_seed(seed)  # seed for weight initialization

  INITMLPERF = getenv("INITMLPERF")
  RUNMLPERF = getenv("RUNMLPERF")
  if getenv("LOGMLPERF"):
    from mlperf_logging import mllog
    import mlperf_logging.mllog.constants as mllog_constants
    mllog.config(filename=f"result_{seed}.txt")
    mllog.config(root_dir=Path(__file__).parents[3].as_posix())  # truncate to log this. "file": "tinygrad/examples/mlperf/model_train.py"
    MLLOGGER = mllog.get_mllogger()
    if INITMLPERF:
      # common.yaml
      MLLOGGER.event(key=mllog_constants.SUBMISSION_ORG, value="tinycorp")
      MLLOGGER.event(key=mllog_constants.SUBMISSION_PLATFORM, value=getenv("SUBMISSION_PLATFORM", "tinybox"))
      MLLOGGER.event(key=mllog_constants.SUBMISSION_DIVISION, value=mllog_constants.CLOSED)
      MLLOGGER.event(key=mllog_constants.SUBMISSION_STATUS, value=mllog_constants.ONPREM)
      # closed_common.yaml
      MLLOGGER.event(key=mllog_constants.SUBMISSION_BENCHMARK, value=mllog_constants.RESNET)
      diskcache_clear()
      MLLOGGER.event(key=mllog_constants.CACHE_CLEAR, value=True)
      MLLOGGER.start(key=mllog_constants.INIT_START)
    if RUNMLPERF:
      MLLOGGER.start(key=mllog_constants.RUN_START)
      MLLOGGER.event(key=mllog_constants.SEED, value=seed)
  else:
    MLLOGGER = None

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"training on {GPUS}")
  for x in GPUS: Device[x]

  TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  EVAL_BEAM = getenv("EVAL_BEAM", BEAM.value)

  # ** model definition and initializers **
  num_classes = 1000
  resnet.Conv2d = Conv2dHeNormal
  resnet.Linear = Linear
  if not getenv("SYNCBN"): resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=len(GPUS))
  model = resnet.ResNet50(num_classes)

  # shard weights and initialize in order
  for k, x in get_state_dict(model).items():
    if not getenv("SYNCBN") and ("running_mean" in k or "running_var" in k):
      x.realize().shard_(GPUS, axis=0)
    else:
      x.realize().to_(GPUS)
  parameters = get_parameters(model)

  # ** hyperparameters **
  epochs            = config["epochs"]            = getenv("EPOCHS", 37)
  BS                = config["BS"]                = getenv("BS", 104 * len(GPUS))  # fp32 GPUS<=6 7900xtx can fit BS=112
  EVAL_BS           = config["EVAL_BS"]           = getenv("EVAL_BS", BS)
  base_lr           = config["base_lr"]           = getenv("LR", 7.2 * (BS/1536))
  lr_warmup_epochs  = config["lr_warmup_epochs"]  = getenv("WARMUP_EPOCHS", 2)
  decay             = config["decay"]             = getenv("DECAY", 2e-4)

  loss_scaler       = config["LOSS_SCALER"]       = getenv("LOSS_SCALER", 128.0 if dtypes.default_float == dtypes.float16 else 1.0)

  target, achieved  = getenv("TARGET", 0.759), False
  eval_start_epoch  = getenv("EVAL_START_EPOCH", 0)
  eval_freq         = getenv("EVAL_FREQ", 1)

  steps_in_train_epoch  = config["steps_in_train_epoch"]  = (round_up(len(get_train_files()), BS) // BS)
  steps_in_val_epoch    = config["steps_in_val_epoch"]    = (round_up(len(get_val_files()), EVAL_BS) // EVAL_BS)

  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["BEAM"]          = BEAM.value
  config["TRAIN_BEAM"]    = TRAIN_BEAM
  config["EVAL_BEAM"]     = EVAL_BEAM
  config["WINO"]          = WINO.value
  config["SYNCBN"]        = getenv("SYNCBN")

  # ** Optimizer **
  skip_list = [v for k, v in get_state_dict(model).items() if "bn" in k or "bias" in k or "downsample.1" in k]
  parameters = [x for x in parameters if x not in set(skip_list)]
  optimizer = LARS(parameters, base_lr, momentum=.9, weight_decay=decay)
  optimizer_skip = SGD(skip_list, base_lr, momentum=.9, weight_decay=0.0, classic=True)
  optimizer_group = OptimizerGroup(optimizer, optimizer_skip)

  # ** LR scheduler **
  scheduler = PolynomialDecayWithWarmup(optimizer, initial_lr=base_lr, end_lr=1e-4,
                                        train_steps=epochs * steps_in_train_epoch,
                                        warmup=lr_warmup_epochs * steps_in_train_epoch)
  scheduler_skip = PolynomialDecayWithWarmup(optimizer_skip, initial_lr=base_lr, end_lr=1e-4,
                                             train_steps=epochs * steps_in_train_epoch,
                                             warmup=lr_warmup_epochs * steps_in_train_epoch)
  scheduler_group = LRSchedulerGroup(scheduler, scheduler_skip)
  print(f"training with batch size {BS} for {epochs} epochs")

  # log mlperf hparams
  if MLLOGGER:
    if RUNMLPERF:
      MLLOGGER.event(key=mllog_constants.GLOBAL_BATCH_SIZE, value=BS)
      from extra.datasets.imagenet import get_train_files, get_val_files
      MLLOGGER.event(key=mllog_constants.TRAIN_SAMPLES, value=len(get_train_files()))
      MLLOGGER.event(key=mllog_constants.EVAL_SAMPLES, value=len(get_val_files()))

      MLLOGGER.event(key=mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=1)
      MLLOGGER.event(key=mllog_constants.OPT_NAME, value="lars")
      assert scheduler.initial_lr == scheduler_skip.initial_lr
      assert scheduler.end_lr == scheduler_skip.end_lr
      assert scheduler.power == scheduler_skip.power
      MLLOGGER.event(key=mllog_constants.LARS_OPT_BASE_LEARNING_RATE, value=scheduler.initial_lr)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_END_LR, value=scheduler.end_lr)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_LR_DECAY_POLY_POWER, value=scheduler.power)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_LR_DECAY_STEPS, value=epochs)
      MLLOGGER.event(key=mllog_constants.LARS_EPSILON, value=0)  # does not support epsilon != 0
      MLLOGGER.event(key=mllog_constants.LARS_OPT_LEARNING_RATE_WARMUP_EPOCHS, value=lr_warmup_epochs)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_MOMENTUM, value=optimizer.momentum)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_WEIGHT_DECAY, value=optimizer.wd)

  # ** resume from checkpointing **
  start_epoch = 0
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer_group, scheduler_group, safe_load(ckpt))
    start_epoch = int(scheduler.epoch_counter.numpy().item() / steps_in_train_epoch)
    print(f"resuming from {ckpt} at epoch {start_epoch}")

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args)

  BENCHMARK = getenv("BENCHMARK")

  # ** jitted steps **
  input_mean = Tensor([123.68, 116.78, 103.94], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  # mlperf reference resnet does not divide by input_std for some reason
  # input_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  def normalize(x): return (x.permute([0, 3, 1, 2]) - input_mean).cast(dtypes.default_float)
  @TinyJit
  def train_step(X, Y):
    optimizer_group.zero_grad()
    X = normalize(X)
    out = model.forward(X)
    loss = out.cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    (loss * loss_scaler).backward()
    for t in optimizer_group.params: t.grad = t.grad.contiguous() / loss_scaler
    optimizer_group.step()
    scheduler_group.step()
    return loss.realize(), top_1.realize()

  @TinyJit
  def eval_step(X, Y):
    X = normalize(X)
    out = model.forward(X)
    loss = out.cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    return loss.realize(), top_1.realize()

  def fake_data_get(batch_size):
    x = Tensor.zeros(batch_size, 224, 224, 3, dtype=dtypes.uchar).contiguous()
    y = [0] * batch_size
    return x.shard(GPUS, axis=0).realize(), Tensor(y, requires_grad=False).shard(GPUS, axis=0), y, None

  def data_get(it):
    x, y, cookie = next(it)
    return x.shard(GPUS, axis=0).realize(), Tensor(y, requires_grad=False).shard(GPUS, axis=0), y, cookie

  # ** epoch loop **
  step_times = []
  for e in range(start_epoch, epochs):
    # ** train loop **
    if MLLOGGER and RUNMLPERF:
      MLLOGGER.start(key=mllog_constants.EPOCH_START, value=e+1, metadata=dict(epoch_num=e+1))
    Tensor.training = True
    BEAM.value = TRAIN_BEAM

    if INITMLPERF:
      i, proc = 0, fake_data_get(BS)
    else:
      batch_loader = batch_load_resnet(batch_size=BS, val=False, shuffle=True, seed=seed*epochs + e, pad_first_batch=True)
      it = iter(tqdm(batch_loader, total=steps_in_train_epoch, desc=f"epoch {e}", disable=BENCHMARK))
      i, proc = 0, data_get(it)

    prev_cookies = []
    st = time.perf_counter()
    while proc is not None:
      GlobalCounters.reset()
      (loss, top_1), y, proc = train_step(proc[0], proc[1]), proc[2], proc[3]

      pt = time.perf_counter()

      if len(prev_cookies) == getenv("STORE_COOKIES", 1): prev_cookies = []  # free previous cookies after gpu work has been enqueued
      try:
        if INITMLPERF:
          next_proc = fake_data_get(BS)
        else:
          next_proc = data_get(it)
      except StopIteration:
        next_proc = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss, top_1 = loss.numpy().item(), top_1.numpy().item()
      top_1_acc = top_1 / sum(yi != -1 for yi in y)

      cl = time.perf_counter()
      if BENCHMARK:
        step_times.append(cl - st)

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {top_1_acc:3.2f} acc, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/top_1_acc": top_1_acc, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": e + (i + 1) / steps_in_train_epoch})

      st = cl
      prev_cookies.append(proc)
      proc, next_proc = next_proc, None  # return old cookie
      i += 1

      if i == BENCHMARK:
        assert not math.isnan(loss)
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_minutes = int(median_step_time * steps_in_train_epoch * epochs / 60)
        print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
        print(f"epoch global_ops: {steps_in_train_epoch * GlobalCounters.global_ops:_}, "
              f"epoch global_mem: {steps_in_train_epoch * GlobalCounters.global_mem:_}")
        # if we are doing beam search, run the first eval too
        if (TRAIN_BEAM or EVAL_BEAM) and e == start_epoch: break
        return
    if MLLOGGER and RUNMLPERF:
      MLLOGGER.event(key=mllog_constants.EPOCH_STOP, value=e+1, metadata=dict(epoch_num=e+1))

    # ** eval loop **
    # always eval for epoch >= 33 to stop the clock as soon as eval target hits, it can converge in epoch in [33, 37]
    if steps_in_val_epoch > 0 and ((e + 1 - eval_start_epoch) % eval_freq == 0 or e + 1 >= 33):
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.start(key=mllog_constants.EVAL_START, value=e+1, metadata=dict(epoch_num=e+1))
      if getenv("RESET_STEP", 1): train_step.reset()  # free the train step memory :(
      eval_times = []
      eval_loss = 0.0
      eval_top_1 = 0
      eval_num_samples = 0
      Tensor.training = False
      BEAM.value = EVAL_BEAM

      if INITMLPERF:
        i, proc = 0, fake_data_get(EVAL_BS)
      else:
        it = iter(tqdm(batch_load_resnet(batch_size=EVAL_BS, val=True, shuffle=False, pad_first_batch=True), total=steps_in_val_epoch))
        i, proc = 0, data_get(it)
        
      prev_cookies = []
      while proc is not None:
        GlobalCounters.reset()
        st = time.time()

        (loss, top_1), y, proc = eval_step(proc[0], proc[1]), proc[2], proc[3]  # drop inputs, keep cookie

        if len(prev_cookies) == getenv("STORE_COOKIES", 1): prev_cookies = []  # free previous cookies after gpu work has been enqueued
        try:
          if INITMLPERF:
            next_proc = fake_data_get(EVAL_BS)
          else:
            next_proc = data_get(it)
        except StopIteration:
          next_proc = None

        loss, top_1 = loss.numpy().item(), top_1.numpy().item()
        num_samples = sum(yi != -1 for yi in y)
        eval_loss += loss * num_samples
        eval_top_1 += top_1
        eval_num_samples += num_samples
        prev_cookies.append(proc)
        proc, next_proc = next_proc, None
        i += 1
        if i == BENCHMARK:
          # assume INITMLPERF has BENCHMARK set
          if MLLOGGER and INITMLPERF:
            MLLOGGER.event(key=mllog_constants.INIT_STOP)
          return

        et = time.time()
        eval_times.append(et - st)

      if getenv("RESET_STEP", 1): eval_step.reset()
      if not BENCHMARK:
        assert eval_num_samples == len(get_val_files()), f"eval sample count mismatched. {eval_num_samples=} != {len(get_val_files())}"
      total_loss = eval_loss / eval_num_samples
      total_top_1 = eval_top_1 / eval_num_samples
      total_fw_time = sum(eval_times) / len(eval_times)
      tqdm.write(f"eval loss: {total_loss:.2f}, eval time: {total_fw_time:.2f}, eval top 1 acc: {total_top_1:.3f}")
      if WANDB:
        wandb.log({"eval/loss": total_loss, "eval/top_1_acc": total_top_1, "eval/forward_time": total_fw_time, "epoch": e + 1})
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.event(key=mllog_constants.EVAL_ACCURACY, value=total_top_1, metadata=dict(epoch_num=e+1))
        MLLOGGER.event(key=mllog_constants.EVAL_STOP, value=e+1, metadata=dict(epoch_num=e+1))

      # save model if achieved target
      if not achieved and total_top_1 >= target:
        # stop once achieve the target
        if MLLOGGER and RUNMLPERF:
          MLLOGGER.event(key=mllog_constants.RUN_STOP, metadata=dict(status=mllog_constants.SUCCESS))
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        fn = f"./ckpts/resnet50_{seed}.safe"
        safe_save(get_state_dict(model), fn)
        print(f" *** Model saved to {fn} ***")
        achieved = True
        break

      # checkpoint every time we eval
      if getenv("CKPT"):
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        if WANDB and wandb.run is not None:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{e}.safe"
        else:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{e}.safe"
        print(f"saving ckpt to {fn}")
        safe_save(get_training_state(model, optimizer_group, scheduler_group), fn)

def train_retinanet():
  from contextlib import redirect_stdout
  import numpy as np
  import math

  SEED = getenv("SEED", 42)
  Tensor.manual_seed(SEED)
  np.random.seed(SEED)
  WANDB = getenv('WANDB')
  HOSTNAME = getenv('SLURM_STEP_NODELIST', 'other')
  EPOCHS = 5
  BS = getenv('BS', 52)
  BS_EVAL = getenv('BS_EVAL', BS)
  WARMUP_EPOCHS = 1
  WARMUP_FACTOR = 0.001
  LR = 0.000085
  MAP_TARGET = 0.34
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  SYNCBN = False
  SYNCBN = True
  TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  EVAL_BEAM = getenv("EVAL_BEAM", BEAM.value)
  loss_scaler = 128.0 if dtypes.default_float in [dtypes.float16] else 1.0

  if WANDB:
    import wandb
    wandb.init(project='RetinaNet')
  
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]
  from extra.models import retinanet
  from examples.mlperf.initializers import Conv2dNormal, Conv2dKaiming

  prior_probability=0.01
  retinanet.Conv2dNormal = Conv2dNormal
  retinanet.Conv2dNormal_prior_prob = functools.partial(Conv2dNormal, b=-math.log((1 - prior_probability) / prior_probability))
  retinanet.Conv2dKaiming = Conv2dKaiming

  from extra.models.retinanet import AnchorGenerator
  anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
  aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
  anchor_generator = AnchorGenerator(
      anchor_sizes, aspect_ratios
  )

  from extra.models import resnet
  from examples.hlb_cifar10 import UnsyncedBatchNorm
  from tinygrad.nn.optim import Adam
  from extra.datasets.openimages_new import get_openimages, batch_load_retinanet, batch_load_retinanet_val
  from pycocotools.cocoeval import COCOeval
  from pycocotools.coco import COCO
  from tinygrad.nn import BatchNorm2d
  ROOT = 'extra/datasets/open-images-v6TEST'
  NAME = 'openimages-mlperf'
  coco_train = get_openimages(NAME,ROOT, 'train')
  coco_val = get_openimages(NAME,ROOT, 'val')
  def cust_call(self, x:Tensor):
    batch_mean = self.running_mean
      # NOTE: this can be precomputed for static inference. we expand it here so it fuses
    batch_invstd = self.running_var.reshape(1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()

    return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd)
  resnet.BatchNorm = functools.partial(BatchNorm2d, eps=0.0)
  resnet.BatchNorm.__call__ = cust_call
  if not SYNCBN: resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=len(GPUS))
  resnet_model = resnet.ResNeXt50_32X4D()
  resnet_model.load_from_pretrained()
  model = retinanet.RetinaNet(resnet_model, num_anchors=anchor_generator.num_anchors_per_location()[0])
  model.backbone.body.fc = None
  # model.load_from_pretrained()

  for k, v in get_state_dict(model).items():
    if 'head' in k and ('clas' in k or 'reg' in k ):
      v.requires_grad = True
    elif k.split('.')[2] in ["layer4", "layer3", "layer2"] and 'bn' not in k and 'weight' in k:
      if 'downsample' in k:
        if 'downsample.0' in k:
          v.requires_grad = True
        else:
          v.requires_grad = False
      else:
        v.requires_grad = True
    elif 'fpn' in k:
      v.requires_grad = True
    else:
      v.requires_grad = False
    if not SYNCBN and ("running_mean" in k or "running_var" in k):
      v.realize().shard_(GPUS, axis=0)
    else:
      v.realize().to_(GPUS)
  
  # model.load_checkpoint("./ckpts/retinanet_4xgpu020_B100_E0_11703.safe")
  # model.load_checkpoint("./ckpts/retinanet_4xgpu018_B16_E0.safe")
  # model.load_from_pretrained()

  parameters = get_parameters(model)
  optimizer = Adam(parameters, lr=LR)

  image_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1,-1,1,1)
  image_mean = Tensor([0.485, 0.456, 0.406], device=GPUS, dtype=dtypes.float32).reshape(1,-1,1,1)
  def normalize(x):
    return (((x.permute((0,3,1,2)) / 255.0) - image_mean)/image_std).cast(dtypes.default_float)

  @TinyJit
  def train_step(X, boxes_temp, labels_temp, matched_idxs):
    Tensor.training = True
    optimizer.zero_grad()
    b,r,c = model(normalize(X), True)
    loss_reg = mdl_reg_loss(r, matched_idxs, boxes_temp)
    loss_class = mdl_class_loss(c, matched_idxs, labels_temp)
    loss = loss_reg+loss_class
    (loss*loss_scaler).backward()
    for t in optimizer.params: t.grad = t.grad.contiguous() / loss_scaler
    optimizer.step()
    return loss.realize()
  @TinyJit
  def val_step(X):
    Tensor.training = False
    b,r,c = model(normalize(X), True)
    out = (r.cat(c.sigmoid(), dim=-1)).cast(dtypes.float32)
    return out.realize()

  feature_shapes = [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]
  ANCHORS = anchor_generator((BS,3,800,800), feature_shapes)
  ANCHORS_STACK = Tensor.stack(ANCHORS)
  ANCHORS_STACK = ANCHORS_STACK.shard(GPUS, axis=0)
  ANCHOR_NP = ANCHORS[0].numpy()
  mdl_reg_loss = lambda r, m, b_t: model.head.regression_head.loss(r,ANCHORS_STACK, m, b_t)
  mdl_class_loss = lambda c, m, l_t: model.head.classification_head.loss(c,m, l_t)

  def data_get(it):
    x, yb, yl, ym, cookie = next(it)
    return x.shard(GPUS, axis=0), yb.shard(GPUS, axis=0), yl.shard(GPUS, axis=0), ym.shard(GPUS, axis=0), cookie 
  def data_get_val(it):
    x, Y_idx, cookie = next(it)
    return x.shard(GPUS, axis=0), Y_idx, cookie

  for epoch in range(EPOCHS):
    print(colored(f'EPOCH {epoch}/{EPOCHS}:', 'cyan'))
    # **********TRAIN***************
    Tensor.training = True
    BEAM.value = TRAIN_BEAM
    batch_loader = batch_load_retinanet(coco_train, bs=BS, seed=SEED, shuffle=False, anchor_np=ANCHOR_NP)
    it = iter(tqdm(batch_loader, total=len(coco_train)//BS, desc=f"epoch {epoch}"))
    cnt, proc = 0, data_get(it)

    st = time.perf_counter()
    while proc is not None:
      GlobalCounters.reset()
      loss, proc = train_step(proc[0], proc[1], proc[2], proc[3]), proc[4]

      pt = time.perf_counter()
      try:
        next_proc = data_get(it)
      except StopIteration:
        next_proc = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss = loss.numpy().item()
      cl = time.perf_counter()

      tqdm.write(
        f"{cnt:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB: wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": epoch + (cnt + 1) / (len(coco_train)//BS)})

      st = cl
      proc, next_proc = next_proc, None  # return old cookie
      cnt+=1


    if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
    fn = f"./ckpts/retinanet_{len(GPUS)}x{HOSTNAME}_B{BS}_E{epoch}.safe"
    state_dict = get_state_dict(model)
    for k,v in state_dict.items():
      state_dict[k] = v.cast(dtypes.float32)
    safe_save(state_dict, fn)
    print(f" *** Model saved to {fn} ***")
    
    # ***********EVAL******************
    bt = time.time()
    if getenv("RESET_STEP", 1): train_step.reset()
    Tensor.training = False
    BEAM.value = EVAL_BEAM
    print(colored(f'{epoch} START EVAL', 'cyan'))
    coco_eval = COCOeval(coco_val.coco, iouType="bbox")
    eval_times = []
    coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(coco_eval.params.catIds), len(coco_eval.params.areaRng)

    batch_loader = batch_load_retinanet_val(coco_val, bs=BS_EVAL, shuffle=False, seed=SEED)
    it = iter(tqdm(batch_loader, total=len(coco_val)//BS_EVAL, desc=f"epoch_val {epoch}"))
    cnt, proc = 0, data_get_val(it)

    while proc is not None:
      GlobalCounters.reset()
      st = time.time()
      y_idxs = proc[1]
      orig_shapes = []
      img_ids = []
      for i in y_idxs:
        img_ids.append(coco_val.ids[i])
        orig_shapes.append(coco_val.__getitem__(i)[0].size[::-1])
      with Tensor.inference_mode():
        out, proc = val_step(proc[0]), proc[2]

      try:
        next_proc = data_get_val(it)
      except StopIteration:
        next_proc = None

      out = out.numpy()
      predictions = model.postprocess_detections(out, orig_image_sizes=orig_shapes)
      coco_results  = [{"image_id": img_ids[i], "category_id": label, "bbox": box.tolist(),
                         "score": score} for i, prediction in enumerate(predictions) 
                         for box, score, label in zip(*prediction.values())]

      # IF COCO_RESULTS LOWER THAN THRESH, ERROR IN EVAL
      # REFERNCE PUSHES EMPTY COCO OBJ
      with redirect_stdout(None):
        coco_eval.cocoDt = coco_val.coco.loadRes(coco_results) if coco_results else COCO()
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
      evaluated_imgs.extend(img_ids)
      coco_evalimgs.append(np.array(coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
      eval_times.append(time.time()-st)
      # print(colored(f'{cnt} EVAL_STEP || {eval_times[-1]}', 'red'))
      cnt=cnt+1
      proc, next_proc = next_proc, None  # return old cookie
      # if cnt>30: break

    coco_eval.params.imgIds = evaluated_imgs
    coco_eval._paramsEval.imgIds = evaluated_imgs
    coco_eval.evalImgs = list(np.concatenate(coco_evalimgs, -1).flatten())
    coco_eval.accumulate()
    coco_eval.summarize()
    eval_acc = coco_eval.stats[0]
    print(colored(f'{epoch} EVAL_ACC {eval_acc} || {time.time()-bt}', 'green'))
    if getenv("RESET_STEP", 1): val_step.reset()

    
    if eval_acc>MAP_TARGET:
      print('SUCCESSFULLY TRAINED TO TARGET: EPOCH', epoch, eval_acc)

      
def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

@TinyJit
def train_step_bert(model, optimizer, scheduler, loss_scaler:float, input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor, masked_lm_ids:Tensor, masked_lm_weights:Tensor, next_sentence_labels:Tensor):
  optimizer.zero_grad()

  lm_logits, seq_relationship_logits = model(input_ids, attention_mask, masked_positions, segment_ids)
  loss = model.loss(lm_logits, seq_relationship_logits, masked_lm_ids, masked_lm_weights, next_sentence_labels)
  (loss * loss_scaler).backward()

  global_norm = Tensor([0.0], dtype=dtypes.float32, device=optimizer[0].device).realize()
  for p in optimizer.params: 
    p.grad = p.grad / loss_scaler
    global_norm += p.grad.float().square().sum()
  global_norm = global_norm.sqrt()
  for p in optimizer.params: p.grad = (p.grad / Tensor.where(global_norm > 1.0, global_norm, 1.0)).cast(p.grad.dtype)

  optimizer.step()
  scheduler.step()
  return loss.realize()

@TinyJit
def eval_step_bert(model, input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor, masked_lm_ids:Tensor, masked_lm_weights:Tensor, next_sentence_labels:Tensor):
  lm_logits, seq_relationship_logits = model(input_ids, attention_mask, masked_positions, segment_ids)
  masked_lm_accuracy, seq_relationship_accuracy, masked_lm_loss, next_sentence_loss = model.accuracy(lm_logits, seq_relationship_logits, masked_lm_ids, masked_lm_weights, next_sentence_labels)
  return {
    "masked_lm_accuracy": masked_lm_accuracy.realize(),
    "next_sentence_accuracy": seq_relationship_accuracy.realize(),
    "masked_lm_loss": masked_lm_loss.realize(),
    "next_sentence_loss": next_sentence_loss.realize()
  }

def train_bert():
  # NOTE: pip install tensorflow, wandb required
  from examples.mlperf.dataloader import batch_load_train_bert, batch_load_val_bert
  from examples.mlperf.helpers import get_mlperf_bert_model, get_data_bert
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup

  config = {}
  BASEDIR = getenv("BASEDIR", Path(__file__).parent.parents[1] / "extra" / "datasets" / "wiki")

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"training on {GPUS}")
  for x in GPUS: Device[x]
  seed = config["seed"] = getenv("SEED", 12345)

  # ** hyperparameters **
  BS                 = config["GLOBAL_BATCH_SIZE"]      = getenv("BS", 16 * len(GPUS) if dtypes.default_float in (dtypes.float16, dtypes.bfloat16) else 8 * len(GPUS))
  EVAL_BS            = config["EVAL_BS"]                = getenv("EVAL_BS", 1 * len(GPUS))
  max_lr             = config["OPT_BASE_LEARNING_RATE"] = getenv("OPT_BASE_LEARNING_RATE", 0.00035 * math.sqrt(BS/256))

  train_steps        = config["TRAIN_STEPS"]            = getenv("TRAIN_STEPS", 4800000 // BS)
  warmup_steps       = config["NUM_WARMUP_STEPS"]       = getenv("NUM_WARMUP_STEPS", 1)
  max_eval_steps     = config["MAX_EVAL_STEPS"]         = getenv("MAX_EVAL_STEPS", (10000 + EVAL_BS - 1) // EVAL_BS) # EVAL_BS * MAX_EVAL_STEPS >= 10000
  eval_step_freq     = config["EVAL_STEP_FREQ"]         = getenv("EVAL_STEP_FREQ", int((math.floor(0.05 * (230.23 * BS + 3000000) / 25000) * 25000) / BS)) # Round down
  save_ckpt_freq     = config["SAVE_CKPT_FREQ"]         = getenv("SAVE_CKPT_FREQ", 1000)
  keep_ckpt_amount   = config["KEEP_CKPT_AMOUNT"]       = getenv("KEEP_CKPT_AMOUNT", 5)
  save_ckpt_dir      = config["SAVE_CKPT_DIR"]          = getenv("SAVE_CKPT_DIR", "./ckpts")
  init_ckpt          = config["INIT_CKPT_DIR"]          = getenv("INIT_CKPT_DIR", BASEDIR)

  loss_scaler        = config["LOSS_SCALER"]            = getenv("LOSS_SCALER", 2.0**9 if dtypes.default_float == dtypes.float16 else 1.0)
  decay              = config["DECAY"]                  = getenv("DECAY", 0.01)
  poly_power         = config["POLY_POWER"]             = getenv("POLY_POWER", 1.0)

  target, achieved                                      = getenv("TARGET", 0.72), False

  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["DISABLE_DROPOUT"] = getenv("DISABLE_DROPOUT", 0)
  config["TRAIN_BEAM"]    = TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  config["EVAL_BEAM"]     = EVAL_BEAM  = getenv("EVAL_BEAM", BEAM.value)

  Tensor.manual_seed(seed)  # seed for weight initialization

  model = get_mlperf_bert_model(init_ckpt)
  
  for _, x in get_state_dict(model).items():
    x.realize().to_(GPUS)
  parameters = get_parameters(model)

  assert 10000 <= (EVAL_BS * max_eval_steps), "Evaluation batchsize * max_eval_steps must greater or equal 10000 to iterate over full eval dataset"

  # ** Log run config **
  for key, value in config.items(): print(f'HParam: "{key}": {value}')

  # ** Optimizer **
  parameters_no_wd = [v for k, v in get_state_dict(model).items() if "bias" in k or "LayerNorm" in k]
  parameters = [x for x in parameters if x not in set(parameters_no_wd)]
  optimizer_wd = LAMB(parameters, lr=max_lr, eps=1e-6, weight_decay=decay, adam=False)
  optimizer_no_wd = LAMB(parameters_no_wd, lr=max_lr, eps=1e-6, weight_decay=0.0, adam=False)
  optimizer_group = OptimizerGroup(optimizer_wd, optimizer_no_wd)

  # ** LR scheduler **
  scheduler_wd = PolynomialDecayWithWarmup(optimizer_wd, max_lr, 0, train_steps, warmup_steps, power=poly_power)
  scheduler_no_wd = PolynomialDecayWithWarmup(optimizer_no_wd, max_lr, 0, train_steps, warmup_steps, power=poly_power)
  scheduler_group = LRSchedulerGroup(scheduler_wd, scheduler_no_wd)
  print(f"training with batch size {BS} for one epoch with {train_steps} steps")

  # ** resume from checkpointing **
  start_step = 0
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer_group, scheduler_group, safe_load(ckpt))
    start_step = int(scheduler_wd.epoch_counter.numpy().item())
    print(f"resuming from {ckpt} at step {start_step}")

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args, project="MLPerf-BERT")

  BENCHMARK = getenv("BENCHMARK")

  eval_it = iter(batch_load_val_bert(EVAL_BS))
  train_it = iter(tqdm(batch_load_train_bert(BS, start_step), initial=start_step, total=train_steps, disable=BENCHMARK))

  step_times = []
  # ** train loop **
  wc_start = time.perf_counter()
  i, train_data = start_step, get_data_bert(GPUS, train_it)
  while train_data is not None and i < train_steps and not achieved:
    Tensor.training = True
    BEAM.value = TRAIN_BEAM
    st = time.perf_counter()
    GlobalCounters.reset()
    loss = train_step_bert(model, optimizer_group, scheduler_group, loss_scaler,
      train_data["input_ids"], train_data["segment_ids"], train_data["input_mask"], train_data["masked_lm_positions"], \
      train_data["masked_lm_ids"], train_data["masked_lm_weights"], train_data["next_sentence_labels"])

    pt = time.perf_counter()

    try:
      next_data = get_data_bert(GPUS, train_it)
    except StopIteration:
      next_data = None

    dt = time.perf_counter()

    device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
    loss = loss.numpy().item()

    cl = time.perf_counter()
    if BENCHMARK: step_times.append(cl - st)

    tqdm.write(
      f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
      f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {optimizer_wd.lr.numpy()[0]:.6f} LR, "
      f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
    if WANDB:
      wandb.log({"lr": optimizer_wd.lr.numpy(), "train/loss": loss, "train/step_time": cl - st,
                  "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                  "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st)})

    train_data, next_data = next_data, None
    i += 1

    if i == BENCHMARK:
      median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
      estimated_total_minutes = int(median_step_time * train_steps / 60)
      print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
      print(f"epoch global_ops: {train_steps * GlobalCounters.global_ops:_}, "
            f"epoch global_mem: {train_steps * GlobalCounters.global_mem:_}")
      return

    # ** eval loop **
    if i % eval_step_freq == 0 or i == 1:
      train_step_bert.reset()  # free the train step memory :(
      eval_lm_losses = []
      eval_clsf_losses = []
      eval_lm_accs = []
      eval_clsf_accs = []
      eval_times = []
      Tensor.training = False
      BEAM.value = EVAL_BEAM

      for j in tqdm(range(max_eval_steps), desc="Evaluating", total=max_eval_steps, disable=BENCHMARK):
        eval_data = get_data_bert(GPUS, eval_it)
        GlobalCounters.reset()
        st = time.time()

        eval_result: dict[str, Tensor] = eval_step_bert(model,
          eval_data["input_ids"], eval_data["segment_ids"], eval_data["input_mask"], eval_data["masked_lm_positions"],
          eval_data["masked_lm_ids"], eval_data["masked_lm_weights"], eval_data["next_sentence_labels"])

        lm_loss, clsf_loss  = eval_result["masked_lm_loss"].item(), eval_result["next_sentence_loss"].item()
        lm_acc, clsf_acc = eval_result["masked_lm_accuracy"].item(), eval_result["next_sentence_accuracy"].item()

        eval_lm_losses.append(lm_loss)
        eval_clsf_losses.append(clsf_loss)
        eval_lm_accs.append(lm_acc)
        eval_clsf_accs.append(clsf_acc)

        et = time.time()
        eval_times.append(et - st)

        if BENCHMARK and j == BENCHMARK: break

      eval_step_bert.reset()
      avg_lm_loss = sum(eval_lm_losses) / len(eval_lm_losses)
      avg_clsf_loss = sum(eval_clsf_losses) / len(eval_clsf_losses)
      avg_lm_acc = sum(eval_lm_accs) / len(eval_lm_accs)
      avg_clsf_acc = sum(eval_clsf_accs) / len(eval_clsf_accs)
      avg_fw_time = sum(eval_times) / len(eval_times)
      results = f"eval lm loss: {avg_lm_loss:.2f}, eval clsf loss: {avg_clsf_loss:.2f}, eval lm accuracy: {avg_lm_acc:.6f}, \
                  eval clsf accuracy: {avg_clsf_acc:.2f}, avg eval step time: {avg_fw_time:.2f}"
      tqdm.write(results)
      with open(getenv("EVAL_LOG", "./eval_log.txt"), "a") as file: file.write(results + "\n")

      if WANDB:
        wandb.log({"eval/lm_loss": avg_lm_loss, "eval/clsf_loss": avg_clsf_loss, "eval/lm_accuracy": avg_lm_acc, \
                    "eval/clsf_accuracy": avg_clsf_acc, "eval/forward_time": avg_fw_time})

      # save model if achieved target
      if not achieved and avg_lm_acc >= target:
        wc_end = time.perf_counter()
        if not os.path.exists(ckpt_dir := save_ckpt_dir): os.mkdir(ckpt_dir)
        fn = f"{ckpt_dir}/bert-large.safe"
        safe_save(get_state_dict(model), fn)
        print(f" *** Model saved to {fn} ***")

        total_seconds = wc_end - wc_start
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        print(f"Reference Convergence point reached after {i * BS} datasamples and {hours}h{minutes}m{seconds:.2f}s.")
        achieved = True
        # stop once hitting the target
        break

    if getenv("CKPT") and i % save_ckpt_freq == 0:
      if not os.path.exists(ckpt_dir := save_ckpt_dir): os.mkdir(ckpt_dir)
      if WANDB and wandb.run is not None:
        fn = f"{ckpt_dir}/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}.safe"
      else:
        fn = f"{ckpt_dir}/{time.strftime('%Y%m%d_%H%M%S')}.safe"
      print(f"saving ckpt to {fn}")
      safe_save(get_training_state(model, optimizer_group, scheduler_group), fn)
      ckpt_files = [f for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f))]
      ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
      while len(ckpt_files) > keep_ckpt_amount:
        last = ckpt_files.pop(0)
        print(f"Removing old ckpt {last}")
        os.remove(os.path.join(ckpt_dir, last))

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  multiprocessing.set_start_method('spawn')
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()