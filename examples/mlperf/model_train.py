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
  
  WANDB = getenv('WANDB')
  config = {}
  SEED            = config['SEED']        = getenv("SEED", 42)
  HOSTNAME        = config['HOST']        = getenv('TINY_TYPE', 'other')
  EPOCHS          = config['EPOCHS']      = getenv('EPOCHS',4)
  BS              = config['BS']          = getenv('BS', 32)
  BS_EVAL         = config['BS_EVAL']     = getenv('BS_EVAL', BS if BS<32 else 32)
  LR              = config['LR']          = 0.000085
  MAP_TARGET      = config['MAP_TARGET']  = 0.34
  GPUS            = config['GPUS']        = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  SYNCBN          = config['SYNCBN']      = getenv("SYNCBN", 0)
  TRAIN_BEAM      = config['TRAIN_BEAM']  = getenv("TRAIN_BEAM", BEAM.value)
  EVAL_BEAM       = config['EVAL_BEAM']   = getenv("EVAL_BEAM", BEAM.value)
  LOSS_SCALAR     = config['LOSS_SCALAR'] = 2048.0 if dtypes.default_float in [dtypes.float16] else 1.0
  TEST            = config['TEST']        = getenv('TEST', 0)
  BENCHMARK       = config['BENCHMARK']   = getenv("BENCHMARK", 10000)
  EVAL_ONLY       = config['EVAL_ONLY']   = getenv('EVAL_ONLY')
  CHKPT_PATH      = config['CHKPT_PATH']  = getenv('CHKPT_PATH', f'ckpts/retinanet_6x{HOSTNAME}_B96_E4.safe')
  TRAIN_ONLY      = config['TRAIN_ONLY']  = getenv('TRAIN_ONLY')
  PART_BATCH      = config['PART_BATCH']  = getenv('PART_BATCH', 1)

  Tensor.manual_seed(SEED)
  np.random.seed(SEED)
  if WANDB:
    import wandb
    wandb.init(project='RetinaNet', config=config)
 
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]

  from extra.models import retinanet
  from extra.models import resnet
  from tinygrad.nn import BatchNorm2d
  from examples.hlb_cifar10 import UnsyncedBatchNorm
  from tinygrad.multi import MultiLazyBuffer
  from examples.mlperf.initializers import Conv2dNormal, Conv2dKaiming, Linear, Conv2dHeNormal
  from tinygrad.nn.optim import Adam
  from examples.mlperf.helpers import anchor_generator
  from extra.datasets.openimages import openimages, get_retinanet_train_files, get_retinanet_val_files
  from examples.mlperf.dataloader import batch_load_retinanet
  from pycocotools.cocoeval import COCOeval
  from pycocotools.coco import COCO

  if not TEST:
    train_files = get_retinanet_train_files()
    val_files = get_retinanet_val_files()

  retinanet.Conv2dNormal = Conv2dNormal
  retinanet.Conv2dNormal_priorprob = functools.partial(Conv2dNormal, b=-math.log(99))
  retinanet.Conv2dKaiming = Conv2dKaiming

  resnet.Conv2d = Conv2dHeNormal
  resnet.Linear = Linear
  if SYNCBN:
    def cust_call_bn(self, x:Tensor):
      batch_mean = self.running_mean
      batch_invstd = self.running_var.reshape(1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()
      return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd).cast(dtypes.default_float)
    resnet.BatchNorm = functools.partial(BatchNorm2d, eps=0.0)
    resnet.BatchNorm.__call__ = cust_call_bn
  else:
    class CustUnsyncedBN(UnsyncedBatchNorm):
      def __call__(self, x:Tensor):
        if isinstance(x.lazydata, MultiLazyBuffer): assert x.lazydata.axis is None or x.lazydata.axis == 0 and len(x.lazydata.lbs) == self.num_devices
        nd = x.shape[0]%self.num_devices
        if nd==0: nd = self.num_devices
        xr = x.reshape(nd, -1, *x.shape[1:]).cast(dtypes.float32)
        batch_mean, batch_invstd = self.calc_stats(xr)
        ret = xr.batchnorm(
          self.weight.reshape(1, -1).expand((nd, -1)),
          self.bias.reshape(1, -1).expand((nd, -1)),
          batch_mean, batch_invstd, axis=(0, 2))
        return ret.reshape(x.shape).cast(x.dtype)
    resnet.BatchNorm = functools.partial(CustUnsyncedBN, num_devices=len(GPUS))
    # resnet.BatchNorm.__call__ = cust_call_unbn
  resnet_model = resnet.ResNeXt50_32X4D()
  resnet_model.load_from_pretrained()

  model = retinanet.RetinaNet(resnet_model)
  model.backbone.body.fc = None

  if EVAL_ONLY and getenv('EVAL_LOAD', 1):
    model.load_checkpoint(CHKPT_PATH)

  for k, v in get_state_dict(model).items():
    if 'head' in k and ('clas' in k or 'reg' in k ): v.requires_grad = True
    elif k.split('.')[2] in ["layer4", "layer3", "layer2"] and 'bn' not in k and 'weight' in k:
      if 'downsample' in k:
        if 'downsample.0' in k: v.requires_grad = True
        else: v.requires_grad = False
      else: v.requires_grad = True
    elif 'fpn' in k: v.requires_grad = True
    else: v.requires_grad = False
    if not SYNCBN and ("running_mean" in k or "running_var" in k): v.realize().shard_(GPUS, axis=0)
    else: v.realize().to_(GPUS)
 
  parameters = get_parameters(model)
  optimizer = Adam(parameters, lr=LR)

  image_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1,-1,1,1)
  image_mean = Tensor([0.485, 0.456, 0.406], device=GPUS, dtype=dtypes.float32).reshape(1,-1,1,1)
  def normalize(x):
    return (((x.permute((0,3,1,2)) / 255.0) - image_mean)/image_std).cast(dtypes.default_float)

  @TinyJit
  def train_step(X, boxes, labels, matched_idxs):
    Tensor.training = True
    optimizer.zero_grad()
    _,r,c = model(normalize(X), True)
    loss_reg = mdl_reg_loss(r.cast(dtypes.float32), matched_idxs, boxes)
    loss_class = mdl_class_loss(c.cast(dtypes.float32), matched_idxs, labels)
    loss = loss_reg+loss_class
    (loss*LOSS_SCALAR).backward()
    for t in optimizer.params: t.grad = t.grad.contiguous() / LOSS_SCALAR
    optimizer.step()
    return loss.realize()

  @TinyJit
  def val_step(X):
    Tensor.training = False
    out = model(normalize(X), False)
    return out.cast(dtypes.float32).to(GPUS[0]).realize()

  feature_shapes = [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]
  ANCHORS = anchor_generator((BS,3,800,800), feature_shapes)
  ANCHORS_STACK = Tensor.stack(*ANCHORS)
  ANCHORS_STACK = ANCHORS_STACK.shard(GPUS, axis=0)
  ANCHOR_NP = ANCHORS[0].numpy()
  mdl_reg_loss = lambda r, m, b_t: model.head.regression_head.loss(r,ANCHORS_STACK, m, b_t)
  mdl_class_loss = lambda c, m, l_t: model.head.classification_head.loss(c,m, l_t)

  def data_get(it):
    x, yb, yl, ym, cookie = next(it)
    return x.shard(GPUS, axis=0), yb.shard(GPUS, axis=0), yl.shard(GPUS, axis=0), ym.shard(GPUS, axis=0), cookie
  def data_get_val(it):
    x, Y_idx, Y_size, cookie = next(it)
    return x.shard(GPUS, axis=0), Y_idx, Y_size, cookie
  def fake_data_get(bs=BS):
    x = Tensor.zeros(bs, 800, 800, 3, dtype=dtypes.uint8).contiguous()
    yb = Tensor.zeros(bs, 120087, 4, dtype=dtypes.float32).contiguous()
    yl = Tensor.ones(bs, 120087, dtype=dtypes.int16).contiguous()
    ym = Tensor.ones(bs, 120087, dtype=dtypes.int64).contiguous()
    return x.shard(GPUS, axis=0), yb.shard(GPUS, axis=0), yl.shard(GPUS, axis=0), ym.shard(GPUS, axis=0), 0

  for epoch in range(1, EPOCHS+1):
    print(colored(f'EPOCH {epoch}/{EPOCHS}:', 'cyan'))
    # **********TRAIN***************
    Tensor.training = True
    BEAM.value = TRAIN_BEAM
    if TEST or EVAL_ONLY:
      cnt, proc = 0, fake_data_get(BS)
    else:
      batch_loader = batch_load_retinanet(batch_size=BS, seed=SEED, shuffle=False, anchor_np=ANCHOR_NP, pad_first_batch=PART_BATCH)
      it = iter(tqdm(batch_loader, total=len(train_files)//BS, desc=f"epoch {epoch}"))
      cnt = 0
      if PART_BATCH:
        train_step.reset()
        pl = round_up(len(train_files), BS) - len(train_files)
        rl = BS - pl
        sl = round_up(rl, len(GPUS)) - len(GPUS)
        x, yb, yl, ym, cookie = next(it)
        proc = x[pl:pl+rl-sl].to(GPUS), yb[pl:pl+rl-sl].to(GPUS), yl[pl:pl+rl-sl].to(GPUS), ym[pl:pl+rl-sl].to(GPUS)
        mdl_reg_loss = lambda r, m, b_t: model.head.regression_head.loss(r,Tensor.stack(*ANCHORS[pl:pl+rl-sl]).to(GPUS), m, b_t)
        loss = train_step(*proc)
        loss_single = loss.numpy().item()
        if WANDB: wandb.log({'train/loss':loss_single})
        print('LOSS_SINGLE', loss_single)
        train_step.reset()
        proc = x[pl+rl-sl:].shard(GPUS, axis=0), yb[pl+rl-sl:].shard(GPUS, axis=0), yl[pl+rl-sl:].shard(GPUS, axis=0), ym[pl+rl-sl:].shard(GPUS, axis=0)
        mdl_reg_loss = lambda r, m, b_t: model.head.regression_head.loss(r,Tensor.stack(*ANCHORS[pl+rl-sl:]).shard(GPUS, axis=0), m, b_t)
        loss = train_step(*proc)
        loss_shard = loss.numpy().item()
        print('LOSS_SHARD', loss_shard)
        if WANDB: wandb.log({'train/loss':loss_shard})
        train_step.reset()
        
        mdl_reg_loss = lambda r, m, b_t: model.head.regression_head.loss(r,ANCHORS_STACK, m, b_t)
        del cookie
      proc = data_get(it)

    st = time.perf_counter()
    bt = st
    while not EVAL_ONLY and proc is not None:
      GlobalCounters.reset()
      loss, proc = train_step(proc[0], proc[1], proc[2], proc[3]), proc[4]

      pt = time.perf_counter()
      try: 
        if TEST: next_proc = fake_data_get(BS)
        else: next_proc = data_get(it)
      except StopIteration: next_proc = None

      dt = time.perf_counter()
      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss = loss.numpy().item()
      cl = time.perf_counter()

      tqdm.write(
        f"{cnt:5} {(cl - st) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB: wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": epoch - 1 + (cnt + 1) / (len(train_files)//BS)})

      st = cl
      proc, next_proc = next_proc, None  # return old cookie
      cnt+=1
      if TEST and cnt>BENCHMARK:
        return
    train_time = time.perf_counter() - bt
    if not EVAL_ONLY:
      print(colored(f'EPOCH {epoch} trained in a total of {train_time / 60:4.2f} minutes', 'red'))
      if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
      fn = f"./ckpts/retinanet_{len(GPUS)}x{HOSTNAME}_B{BS}_E{epoch}.safe"
      state_dict = get_state_dict(model)
      safe_save(state_dict, fn)
      print(f" *** Model saved to {fn} ***")

    # ***********EVAL******************
    if not TRAIN_ONLY:
      bt = time.perf_counter()
      if getenv("RESET_STEP", 1): train_step.reset()
      Tensor.training = False
      BEAM.value = EVAL_BEAM
      print(colored(f'{epoch} START EVAL', 'cyan'))
      coco_val = COCO(openimages('validation'))
      coco_eval = COCOeval(coco_val, iouType="bbox")
      eval_times = []
      coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(coco_eval.params.catIds), len(coco_eval.params.areaRng)

      batch_loader = batch_load_retinanet(batch_size=BS_EVAL, shuffle=False, seed=SEED, val=True)
      it = iter(tqdm(batch_loader, total=len(val_files)//BS_EVAL, desc=f"epoch_val {epoch}"))
      cnt, proc = 0, data_get_val(it)

      while proc is not None:
        cnt+=1
        GlobalCounters.reset()
        st = time.perf_counter()
        
        img_ids = proc[1]
        orig_shapes = proc[2]
        with Tensor.inference_mode():
          out, proc = val_step(proc[0]), proc[3]
        pt = time.perf_counter()

        try: next_proc = data_get_val(it)
        except StopIteration: next_proc = None
        nt = time.perf_counter()

        out = out.numpy(False)
        npt = time.perf_counter()
        
        predictions = model.postprocess_detections(out, orig_image_sizes=orig_shapes)
        coco_results  = [{"image_id": img_ids[i], "category_id": label, "bbox": box.tolist(),
                          "score": score} for i, prediction in enumerate(predictions)
                          for box, score, label in zip(*prediction.values())]
        dt = time.perf_counter()

        with redirect_stdout(None):
          coco_eval.cocoDt = coco_val.loadRes(coco_results) if coco_results else COCO()
          coco_eval.params.imgIds = img_ids
          coco_eval.evaluate()
        ct = time.perf_counter()

        evaluated_imgs.extend(img_ids)
        coco_evalimgs.append(np.array(coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
        eval_times.append(time.time()-st)
        proc, next_proc = next_proc, None
        
        tqdm.write(
          f"{cnt:5} {(ct - st) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms model, {(dt - npt) * 1000.0:7.2f} ms postproc, "
          f"{(nt - pt) * 1000.0:6.2f} ms fetch data, {(npt - nt) * 1000.0:7.2f} ms np, {(ct - dt) * 1000.0:7.2f} ms eval, "
          f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9/(pt-st):9.2f} GFLOPS")
        if WANDB: wandb.log({"eval/step_time": ct - st, "eval/model_time": pt - st, "eval/post_proc": dt - npt, "eval/data": nt - pt, 
                             "eval/np": npt - nt, "eval/evaluate": ct - dt, "eval/GFLOPS": GlobalCounters.global_ops * 1e-9 / (pt - st), 
                             "epoch": epoch - 1 + (cnt + 1) / (len(val_files)//BS_EVAL)})

      coco_eval.params.imgIds = evaluated_imgs
      coco_eval._paramsEval.imgIds = evaluated_imgs
      coco_eval.evalImgs = list(np.concatenate(coco_evalimgs, -1).flatten())
      coco_eval.accumulate()
      coco_eval.summarize()
      eval_acc = coco_eval.stats[0]
      eval_time = time.perf_counter()-bt
      print(colored(f'{epoch} EVAL_ACC {eval_acc} || {eval_time / 60:4.2f}', 'green'))
      if WANDB:
          wandb.log({"eval/acc": eval_acc, "eval/total_time": eval_time, "epoch": epoch})
      if getenv("RESET_STEP", 1): val_step.reset()

      if eval_acc>MAP_TARGET:
        print('SUCCESSFULLY TRAINED TO TARGET: EPOCH', epoch, eval_acc)
        break

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
  from examples.mlperf.helpers import get_mlperf_bert_model, get_data_bert, get_fake_data_bert
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup

  config = {}
  BASEDIR = getenv("BASEDIR", Path(__file__).parent.parents[1] / "extra" / "datasets" / "wiki")

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"training on {GPUS}")
  for x in GPUS: Device[x]
  seed = config["seed"] = getenv("SEED", 12345)

  INITMLPERF = getenv("INITMLPERF")
  RUNMLPERF = getenv("RUNMLPERF")
  if getenv("LOGMLPERF"):
    from mlperf_logging import mllog
    import mlperf_logging.mllog.constants as mllog_constants

    mllog.config(filename="bert.log")
    mllog.config(root_dir=Path(__file__).parents[3].as_posix())
    MLLOGGER = mllog.get_mllogger()
    MLLOGGER.logger.propagate = False

    if INITMLPERF:
      assert BENCHMARK, f"BENCHMARK must be set for INITMLPERF"
      MLLOGGER.event(key=mllog_constants.SUBMISSION_ORG, value="tinycorp")
      MLLOGGER.event(key=mllog_constants.SUBMISSION_PLATFORM, value=getenv("SUBMISSION_PLATFORM", "tinybox"))
      MLLOGGER.event(key=mllog_constants.SUBMISSION_DIVISION, value=mllog_constants.CLOSED)
      MLLOGGER.event(key=mllog_constants.SUBMISSION_STATUS, value=mllog_constants.ONPREM)

      MLLOGGER.event(key=mllog_constants.SUBMISSION_BENCHMARK, value=mllog_constants.BERT)

      diskcache_clear()
      MLLOGGER.event(key=mllog_constants.CACHE_CLEAR, value=True)
      MLLOGGER.start(key=mllog_constants.INIT_START, value=None)

    if RUNMLPERF:
      MLLOGGER.start(key=mllog_constants.RUN_START, value=None)
  else:
    MLLOGGER = None

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
  epsilon            = config["EPSILON"]                = getenv("EPSILON", 1e-6)
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
  optimizer_wd = LAMB(parameters, lr=max_lr, eps=epsilon, weight_decay=decay, adam=False)
  optimizer_no_wd = LAMB(parameters_no_wd, lr=max_lr, eps=epsilon, weight_decay=0.0, adam=False)
  optimizer_group = OptimizerGroup(optimizer_wd, optimizer_no_wd)

  # ** LR scheduler **
  scheduler_wd = PolynomialDecayWithWarmup(optimizer_wd, max_lr, 0, train_steps, warmup_steps, power=poly_power)
  scheduler_no_wd = PolynomialDecayWithWarmup(optimizer_no_wd, max_lr, 0, train_steps, warmup_steps, power=poly_power)
  scheduler_group = LRSchedulerGroup(scheduler_wd, scheduler_no_wd)
  print(f"training with batch size {BS} for one epoch with {train_steps} steps")

  # log mlperf hparams
  if MLLOGGER:
    if RUNMLPERF:
      MLLOGGER.event(key=mllog_constants.GLOBAL_BATCH_SIZE, value=config["GLOBAL_BATCH_SIZE"])
      MLLOGGER.event(key=mllog_constants.MAX_SEQUENCE_LENGTH, value=512)
      MLLOGGER.event(key="max_predictions_per_seq", value=76)

      MLLOGGER.event(key=mllog_constants.OPT_NAME, value="LAMB")
      MLLOGGER.event(key=mllog_constants.OPT_BASE_LR, value=config["OPT_BASE_LEARNING_RATE"])
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_WEIGHT_DECAY, value=config["DECAY"])
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_BETA_1, value=optimizer_wd.b1)
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_BETA_2, value=optimizer_wd.b2)
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=config["POLY_POWER"])
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_EPSILON, value=config["EPSILON"])

      MLLOGGER.event(key=mllog_constants.OPT_LR_WARMUP_STEPS, value=config["NUM_WARMUP_STEPS"])
      MLLOGGER.event(key=mllog_constants.NUM_WARMUP_STEPS, value=config["NUM_WARMUP_STEPS"])
      MLLOGGER.event(key='start_warmup_step', value=0)
      MLLOGGER.event(key='opt_learning_rate_training_steps', value=config["TRAIN_STEPS"])
      MLLOGGER.event(key=mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=1)
      MLLOGGER.event(key=mllog_constants.EVAL_SAMPLES, value=config["EVAL_BS"] * config["MAX_EVAL_STEPS"])
      MLLOGGER.event(key=mllog_constants.TRAIN_SAMPLES, value=config["GLOBAL_BATCH_SIZE"] * config["TRAIN_STEPS"])

  # ** resume from checkpointing **
  start_step = 1
  previous_step = None
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

  if not INITMLPERF:
    eval_it = iter(batch_load_val_bert(EVAL_BS))
    train_it = iter(tqdm(batch_load_train_bert(BS, start_step), initial=start_step, total=train_steps, disable=BENCHMARK))

  step_times = []
  # ** train loop **
  wc_start = time.perf_counter()
  if INITMLPERF:
    i, train_data = start_step, get_fake_data_bert(GPUS, BS)
  else:
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
      if INITMLPERF:
        next_data = get_fake_data_bert(GPUS, BS)
      else:
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

    # ** eval loop **
    if i % eval_step_freq == 0 or (BENCHMARK and i == BENCHMARK):
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.start(key=mllog_constants.EVAL_START, value=None, metadata={"epoch_num": 1, "epoch_count": 1, "step_num": i})
      train_step_bert.reset()
      eval_lm_losses = []
      eval_clsf_losses = []
      eval_lm_accs = []
      eval_clsf_accs = []
      eval_times = []
      Tensor.training = False
      BEAM.value = EVAL_BEAM

      for j in tqdm(range(max_eval_steps), desc="Evaluating", total=max_eval_steps, disable=BENCHMARK):
        if INITMLPERF:
          eval_data = get_fake_data_bert(GPUS, EVAL_BS)
        else:
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

        if BENCHMARK and j == BENCHMARK:
          # assume INITMLPERF has BENCHMARK set
          if MLLOGGER and INITMLPERF:
            MLLOGGER.event(key=mllog_constants.INIT_STOP, value=None)
          return

      eval_step_bert.reset()
      avg_lm_loss = sum(eval_lm_losses) / len(eval_lm_losses)
      avg_clsf_loss = sum(eval_clsf_losses) / len(eval_clsf_losses)
      avg_lm_acc = sum(eval_lm_accs) / len(eval_lm_accs)
      avg_clsf_acc = sum(eval_clsf_accs) / len(eval_clsf_accs)
      avg_fw_time = sum(eval_times) / len(eval_times)
      results = f"eval lm loss: {avg_lm_loss:.2f}, eval clsf loss: {avg_clsf_loss:.2f}, eval lm accuracy: {avg_lm_acc:.6f}, \
                  eval clsf accuracy: {avg_clsf_acc:.2f}, avg eval step time: {avg_fw_time:.2f}"
      tqdm.write(results)

      if WANDB:
        wandb.log({"eval/lm_loss": avg_lm_loss, "eval/clsf_loss": avg_clsf_loss, "eval/lm_accuracy": avg_lm_acc, \
                    "eval/clsf_accuracy": avg_clsf_acc, "eval/forward_time": avg_fw_time})

      if MLLOGGER and RUNMLPERF:
        MLLOGGER.end(key=mllog_constants.EVAL_STOP, value=i, metadata={"epoch_count": 1, "step_num": i, "samples_count": config["EVAL_BS"] * config["MAX_EVAL_STEPS"]})
        MLLOGGER.event(key=mllog_constants.EVAL_ACCURACY, value=avg_lm_acc, metadata={"epoch_num": 1, "masked_lm_accuracy": avg_lm_acc})

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
        if MLLOGGER and RUNMLPERF:
          MLLOGGER.end(key=mllog_constants.RUN_STOP, metadata=dict(status=mllog_constants.SUCCESS))
        # stop once hitting the target
        break

    if getenv("CKPT", 1) and i % save_ckpt_freq == 0:
      if MLLOGGER and RUNMLPERF:
        if previous_step:
          MLLOGGER.end(key=mllog_constants.BLOCK_STOP, value=None, metadata={"first_epoch_num": 1, "epoch_num": 1, "first_step_num": i, "step_num": i, "step_count": i - previous_step})
        MLLOGGER.start(key="checkpoint_start", value=None, metadata={"step_num" : i})
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
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.end(key="checkpoint_stop", value=None, metadata={"step_num": i})
        MLLOGGER.start(key=mllog_constants.BLOCK_START, value=None, metadata={"first_epoch_num": 1, "epoch_num": 1, "epoch_count": 1, "samples_count": config["EVAL_BS"] * config["MAX_EVAL_STEPS"], "step_num": i, "first_step_num": i+1})
        previous_step = i

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