import os, random, pickle, functools, itertools
from typing import List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, prod, Context, round_up
from collections import deque
from multiprocessing import Queue, Process, shared_memory, connection, Lock, cpu_count, Pool

class MyQueue:
  def __init__(self, multiple_readers=True, multiple_writers=True):
    self._reader, self._writer = connection.Pipe(duplex=False)
    self._rlock = Lock() if multiple_readers else None
    self._wlock = Lock() if multiple_writers else None
  def get(self):
    if self._rlock: self._rlock.acquire()
    ret = pickle.loads(self._reader.recv_bytes())
    if self._rlock: self._rlock.release()
    return ret
  def put(self, obj):
    if self._wlock: self._wlock.acquire()
    self._writer.send_bytes(pickle.dumps(obj))
    if self._wlock: self._wlock.release()

def shuffled_indices(n, seed=None):
  rng = random.Random(seed)
  indices = {}
  for i in range(n-1, -1, -1):
    j = rng.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

def loader_process(q_in, q_out, X:Tensor, seed):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  from extra.datasets.imagenet import center_crop, preprocess_train

  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, fn, val = _recv
      if fn is not None:
        img = Image.open(fn)
        img = img.convert('RGB') if img.mode != "RGB" else img

        if val:
          # eval: 76.08%, load in 0m7.366s (0m5.301s with simd)
          # sudo apt-get install libjpeg-dev
          # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
          img = center_crop(img)
          img = np.array(img)
        else:
          # reseed rng for determinism
          if seed is not None:
            np.random.seed(seed * 2 ** 10 + idx)
            random.seed(seed * 2 ** 10 + idx)
          img = preprocess_train(img)
      else:
        # pad data with training mean
        img = np.tile(np.array([[[123.68, 116.78, 103.94]]], dtype=np.uint8), (224, 224, 1))

      # broken out
      #img_tensor = Tensor(img.tobytes(), device='CPU')
      #storage_tensor = X[idx].contiguous().realize().lazydata.realized
      #storage_tensor._copyin(img_tensor.numpy())

      # faster
      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()

      # ideal
      #X[idx].assign(img.tobytes())   # NOTE: this is slow!
      q_out.put(idx)
    q_out.put(None)

def batch_load_resnet(batch_size=64, val=False, shuffle=True, seed=None, pad_first_batch=False):
  from extra.datasets.imagenet import get_train_files, get_val_files
  files = get_val_files() if val else get_train_files()
  from extra.datasets.imagenet import get_imagenet_categories
  cir = get_imagenet_categories()

  if pad_first_batch:
    FIRST_BATCH_PAD = round_up(len(files), batch_size) - len(files)
  else:
    FIRST_BATCH_PAD = 0
  file_count = FIRST_BATCH_PAD + len(files)
  BATCH_COUNT = min(32, file_count // batch_size)

  def _gen():
    for _ in range(FIRST_BATCH_PAD): yield -1
    yield from shuffled_indices(len(files), seed=seed) if shuffle else iter(range(len(files)))
  gen = iter(_gen())

  def enqueue_batch(num):
    for idx in range(num*batch_size, (num+1)*batch_size):
      fidx = next(gen)
      if fidx != -1:
        fn = files[fidx]
        q_in.put((idx, fn, val))
        Y[idx] = cir[fn.split("/")[-2]]
      else:
        # padding
        q_in.put((idx, None, val))
        Y[idx] = -1

  shutdown = False
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while 1:
      num = q_out.get()//batch_size
      gotten[num] += 1
      if gotten[num] == batch_size: break
    gotten[num] = 0
    return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size], Cookie(num)

  #q_in, q_out = MyQueue(multiple_writers=False), MyQueue(multiple_readers=False)
  q_in, q_out = Queue(), Queue()

  sz = (batch_size*BATCH_COUNT, 224, 224, 3)
  if os.path.exists("/dev/shm/resnet_X"): os.unlink("/dev/shm/resnet_X")
  shm = shared_memory.SharedMemory(name="resnet_X", create=True, size=prod(sz))
  procs = []

  try:
    # disk:shm is slower
    #X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/resnet_X")
    Y = [None] * (batch_size*BATCH_COUNT)

    for _ in range(cpu_count()):
      p = Process(target=loader_process, args=(q_in, q_out, X, seed))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    # NOTE: this is batch aligned, last ones are ignored unless pad_first_batch is True
    for _ in range(0, file_count//batch_size): yield receive_batch()
  finally:
    shutdown = True
    # empty queues
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    # shutdown processes
    for p in procs: p.join()
    shm.close()
    try:
      shm.unlink()
    except FileNotFoundError:
      # happens with BENCHMARK set
      pass

@functools.lru_cache(maxsize=128)
def load_bert_file(fn:str) -> List[dict]:
  with open(fn, "rb") as f: data = pickle.load(f)
  return data

def process_batch_bert(data: List[dict]) -> dict[str, Tensor]:
  return {
    "input_ids": Tensor(np.concatenate([s["input_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "input_mask": Tensor(np.concatenate([s["input_mask"] for s in data], axis=0), dtype=dtypes.default_float),
    "segment_ids": Tensor(np.concatenate([s["segment_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_positions": Tensor(np.concatenate([s["masked_lm_positions"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_ids": Tensor(np.concatenate([s["masked_lm_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_weights": Tensor(np.concatenate([s["masked_lm_weights"] for s in data], axis=0), dtype=dtypes.float32),
    "next_sentence_labels": Tensor(np.concatenate([s["next_sentence_labels"] for s in data], axis=0), dtype=dtypes.float32),
  }

def shuffle_parts(file_paths: List[str]) -> List[str]:
  parts = {}
  for f in file_paths:
    part = Path(f).stem.split('_')[0]
    if part not in parts: parts[part] = []
    parts[part].append(f)
  
  part_ids = list(parts.keys())
  random.shuffle(part_ids)

  shuffled_files = []
  for p in part_ids:
    parts[p].sort(key=lambda x: int(Path(x).stem.split('_')[1]))
    shuffled_files.extend(parts[p])
  return shuffled_files

def random_sample(data: List[str]):
  index = random.randint(0, len(data) - 1)
  selected_sample = data[index]
  return selected_sample, index

def load_datasample(file_and_offset:Tuple[str, int]) -> List[dict]:
  data = load_bert_file(file_and_offset[0])
  return data[file_and_offset[1]]

# Reference: https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/run_pretraining.py, Line 394
def batch_load_train_bert(BS:int, start_step:int = 0):
  from extra.datasets.wikipedia import get_wiki_train_files
  files = shuffle_parts(get_wiki_train_files())
  dataset = []
  for f in tqdm(files, desc="Building dataset"):
    lists = [(f, o) for o in range(int(Path(f).stem.split("_")[3].split(".")[0]))]
    dataset.extend(lists)
  
  dataset = dataset[start_step:]
  
  active_set = deque(dataset[:1000])
  remaining_set = deque(dataset[1000:])

  while dataset:
    blob = []
    for _ in range(BS):
      if active_set:
        index = random.randint(0, len(active_set) - 1)
        sample = active_set[index]
        active_set.remove(sample)
        blob.append(sample)
        if remaining_set:
            active_set.append(remaining_set.popleft())
    yield process_batch_bert([load_datasample(sample) for sample in blob])

# Reference: https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/run_pretraining.py, Line 416
def batch_load_val_bert(BS:int):
  from extra.datasets.wikipedia import get_wiki_val_files
  files = get_wiki_val_files()
  dataset = list(itertools.chain.from_iterable([load_bert_file(f) for f in files]))
  idx = 0
  while True:
    start_idx = (idx * BS) % len(dataset)
    end_idx = ((idx + 1) * BS) % len(dataset)
    if start_idx < end_idx:
        yield process_batch_bert(dataset[start_idx:end_idx])
    else:  # wrap around the end to the beginning of the dataset
        yield process_batch_bert(dataset[start_idx:] + dataset[:end_idx])
    idx += 1

def load_unet3d_data(preprocessed_dataset_dir, seed, queue_in, queue_out, X:Tensor, Y:Tensor):
  from extra.datasets.kits19 import rand_balanced_crop, rand_flip, random_brightness_augmentation, gaussian_noise

  while (data := queue_in.get()) is not None:
    idx, fn, val = data
    case_name = os.path.basename(fn).split("_x.npy")[0]
    x, y = np.load(preprocessed_dataset_dir / f"{case_name}_x.npy"), np.load(preprocessed_dataset_dir / f"{case_name}_y.npy")

    if not val:
      if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

      x, y = rand_balanced_crop(x, y)
      x, y = rand_flip(x, y)
      x, y = x.astype(np.float32), y.astype(np.uint8)
      x = random_brightness_augmentation(x)
      x = gaussian_noise(x)

    X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = x.tobytes()
    Y[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = y.tobytes()

    queue_out.put(idx)
  queue_out.put(None)

def batch_load_unet3d(preprocessed_dataset_dir:Path, batch_size:int=6, val:bool=False, shuffle:bool=True, seed=None):
  assert preprocessed_dataset_dir is not None, "run preprocess_data on kits19"

  files = sorted(list(preprocessed_dataset_dir.glob("*_x.npy")))
  file_indices = list(range(len(files)))
  batch_count = min(32, len(files) // batch_size)

  queue_in, queue_out = Queue(), Queue()
  procs, data_out_count = [], [0] * batch_count
  shm_name_x, shm_name_y = "unet3d_x", "unet3d_y"
  sz = (batch_size * batch_count, 1, 128, 128, 128)
  if os.path.exists(f"/dev/shm/{shm_name_x}"): os.unlink(f"/dev/shm/{shm_name_x}")
  if os.path.exists(f"/dev/shm/{shm_name_y}"): os.unlink(f"/dev/shm/{shm_name_y}")
  shm_x = shared_memory.SharedMemory(name=shm_name_x, create=True, size=prod(sz))
  shm_y = shared_memory.SharedMemory(name=shm_name_y, create=True, size=prod(sz))

  shutdown = False
  class Cookie:
    def __init__(self, bc):
      self.bc = bc
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.bc)
        except StopIteration: pass

  def enqueue_batch(bc):
    for idx in range(bc * batch_size, (bc+1) * batch_size):
      fn = files[next(ds_iter)]
      queue_in.put((idx, fn, val))

  def shuffle_indices(file_indices, seed=None):
    rng = random.Random(seed)
    rng.shuffle(file_indices)

  if shuffle: shuffle_indices(file_indices, seed=seed)
  ds_iter = iter(file_indices)

  try:
    X = Tensor.empty(*sz, dtype=dtypes.float32, device=f"disk:/dev/shm/{shm_name_x}")
    Y = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/{shm_name_y}")

    for _ in range(cpu_count()):
      proc = Process(target=load_unet3d_data, args=(preprocessed_dataset_dir, seed, queue_in, queue_out, X, Y))
      proc.daemon = True
      proc.start()
      
      procs.append(proc)

    for bc in range(batch_count):
      enqueue_batch(bc)

    for _ in range(len(files) // batch_size):
      while True:
        bc = queue_out.get() // batch_size
        data_out_count[bc] += 1
        if data_out_count[bc] == batch_size: break

      data_out_count[bc] = 0
      yield X[bc * batch_size:(bc + 1) * batch_size], Y[bc * batch_size:(bc + 1) * batch_size], Cookie(bc)
  finally:
    shutdown = True

    for _ in procs: queue_in.put(None)
    queue_in.close()

    for _ in procs:
      while queue_out.get() is not None: pass
    queue_out.close()

    # shutdown processes
    for proc in procs: proc.join()

    shm_x.close()
    shm_y.close()
    try:
      shm_x.unlink()
      shm_y.unlink()
    except FileNotFoundError:
      # happens with BENCHMARK set
      pass
def loader_process_retinanet(q_in, q_out, seed, X:Tensor, YB:Tensor, YL:Tensor, YM:Tensor, anchor_np):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  from extra.datasets.openimages import preprocess_train
  from examples.mlperf.helpers import matcher_iou_func, resize_boxes_np
  
  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, fn, data = _recv
      if fn is not None:
        img = Image.open(fn)
        img = img.convert('RGB') if img.mode != "RGB" else img
        w, h = img.size
        img = preprocess_train(img)
        
        boxes = [obj for obj in data['bbox']]
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
        classes = [obj for obj in data['CatID']]
        classes = np.array(classes, dtype=np.int16)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        boxes = resize_boxes_np(boxes, (h, w), (800,800))
        random.seed(seed * 2 ** 20 + idx)
        r = random.random() < 0.5
        if r:
          img = np.flip(img, axis=1)
          boxes[:, [0, 2]] = 800 - boxes[:, [2, 0]]
        midx = matcher_iou_func(boxes, anchor_np)
        m_temp = np.clip(midx, 0, None)
        tb = boxes[m_temp]
        tl = classes[m_temp]

      else:
        # pad data with training mean
        img = np.tile(np.array([[[123.68, 116.78, 103.94]]], dtype=np.uint8), (800, 800, 1))
        tb = np.zeros((120087, 4)).astype(np.float32)
        tl = np.zeros((120087)).astype(np.int16)+1
        midx = np.zeros((120087)).astype(np.int64)
      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()
      YB[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = tb.tobytes()
      YL[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = tl.tobytes()
      YM[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = midx.tobytes()

      q_out.put(idx)
    q_out.put(None)
def loader_process_retinanet_val(q_in, q_out, seed, X:Tensor):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  from extra.datasets.openimages import preprocess_train
  
  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, fn = _recv
      if fn is not None:
        img = Image.open(fn)
        img = img.convert('RGB') if img.mode != "RGB" else img
        img = preprocess_train(img)

      else:
        # pad data with training mean
        img = np.tile(np.array([[[123.68, 116.78, 103.94]]], dtype=np.uint8), (800, 800, 1))

      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()

      q_out.put(idx)
    q_out.put(None)

def batch_load_retinanet(batch_size=64, val=False, shuffle=False, seed=42, pad_first_batch=False, anchor_np=[1,2,3,4]):
  from extra.datasets.openimages import get_retinanet_train_files, get_retinanet_val_files, get_train_data, get_val_data
  files = get_retinanet_val_files() if val else get_retinanet_train_files()
  data_dict = get_val_data() if val else get_train_data()
  if val:
    path_img_dict = {}
    for item in data_dict['images']:
      path_img_dict[item['file_name']] = [item['id'], (item['height'], item['width'])]
  if pad_first_batch:
    FIRST_BATCH_PAD = round_up(len(files), batch_size) - len(files)
  else:
    FIRST_BATCH_PAD = 0
  file_count = FIRST_BATCH_PAD + len(files)
  BATCH_COUNT = min(32, file_count // batch_size)

  def _gen():
    for _ in range(FIRST_BATCH_PAD): yield -1
    yield from shuffled_indices(len(files), seed=seed) if shuffle else iter(range(len(files)))
  gen = iter(_gen())

  def enqueue_batch(num):
    for idx in range(num*batch_size, (num+1)*batch_size):
      fidx = next(gen)
      if fidx != -1:
        fn = files[fidx]
        if val:
          q_in.put((idx, fn))
          img_data = path_img_dict[fn.split('/')[-1]]
          Y[idx] = img_data[0]
          YS[idx] = img_data[1]
        else:
          img_path = fn.split('/')[-1].split('.')[0]
          q_in.put((idx, fn, data_dict[img_path]))
      else:
        # padding
        if val:
          q_in.put((idx, None))
        else:
          q_in.put((idx, None, None))

  shutdown = False
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch_train():
    while 1:
      num = q_out.get()//batch_size
      gotten[num] += 1
      if gotten[num] == batch_size: break
    gotten[num] = 0
    return X[num*batch_size:(num+1)*batch_size], YB[num*batch_size:(num+1)*batch_size], \
      YL[num*batch_size:(num+1)*batch_size], YM[num*batch_size:(num+1)*batch_size], Cookie(num)
  def receive_batch_val():
    while 1:
      num = q_out.get()//batch_size
      gotten[num] += 1
      if gotten[num] == batch_size: break
    gotten[num] = 0
    return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size], YS[num*batch_size:(num+1)*batch_size], Cookie(num)

  receive_batch = receive_batch_val if val else receive_batch_train
  q_in, q_out = Queue(), Queue()

  sz = (batch_size*BATCH_COUNT, 800, 800, 3)
  Path('/dev/shm/retinanet_X').unlink(True)
  shm = shared_memory.SharedMemory(name="retinanet_X", create=True, size=prod(sz))
  if not val:
    bsz = (batch_size*BATCH_COUNT, 120087, 4)
    Path('/dev/shm/retinanet_YB').unlink(True)
    bshm = shared_memory.SharedMemory(name="retinanet_YB", create=True, size=prod(bsz))
    lsz = (batch_size*BATCH_COUNT, 120087)
    Path('/dev/shm/retinanet_YL').unlink(True)
    lshm = shared_memory.SharedMemory(name="retinanet_YL", create=True, size=prod(lsz))
    msz = (batch_size*BATCH_COUNT, 120087)
    Path('/dev/shm/retinanet_YM').unlink(True)
    mshm = shared_memory.SharedMemory(name="retinanet_YM", create=True, size=prod(msz))
  procs = []

  try:
    # disk:shm is slower
    #X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/retinanet_X")
    if not val:
      YB = Tensor.empty(*bsz, dtype=dtypes.float32, device=f"disk:/dev/shm/retinanet_YB")
      YL = Tensor.empty(*lsz, dtype=dtypes.int16, device=f"disk:/dev/shm/retinanet_YL")
      YM = Tensor.empty(*msz, dtype=dtypes.int64, device=f"disk:/dev/shm/retinanet_YM")
    else:
      Y = [None] * (batch_size*BATCH_COUNT)
      YS = [None] * (batch_size*BATCH_COUNT)
    for _ in range(10 if val else cpu_count()):
      if val:
        p = Process(target=loader_process_retinanet_val, args=(q_in, q_out, seed, X))
      else:
        p = Process(target=loader_process_retinanet, args=(q_in, q_out, seed, X, YB, YL, YM, anchor_np))
      p.daemon = True
      p.start()
      procs.append(p)
    if FIRST_BATCH_PAD>0:
      enqueue_batch(0)
      start_idx = 1
      yield receive_batch()
    else:
      start_idx = 0

    for bn in range(start_idx, BATCH_COUNT): enqueue_batch(bn)

    # NOTE: this is batch aligned, last ones are ignored unless pad_first_batch is True
    for _ in range(start_idx, file_count//batch_size): yield receive_batch()
  finally:
    shutdown = True
    # empty queues
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    # shutdown processes
    for p in procs: p.join()
    shm.close()
    if not val:
      bshm.close()
      lshm.close()
      mshm.close()
    try:
      shm.unlink()
      if not val:
        bshm.unlink()
        lshm.unlink()
        mshm.unlink()
    except FileNotFoundError:
      # happens with BENCHMARK set
      pass
if __name__ == "__main__":
  def load_unet3d(val):
    assert not val, "validation set is not supported due to different sizes on inputs"

    from extra.datasets.kits19 import get_train_files, get_val_files, preprocess_dataset, BASEDIR
    preprocessed_dataset_dir = (BASEDIR / ".." / "preprocessed" / ("val" if val else "train"))
    files = get_val_files() if val else get_train_files()

    if not preprocessed_dataset_dir.exists(): preprocess_dataset(files, preprocessed_dataset_dir, val)
    with tqdm(total=len(files)) as pbar:
      for x, _, _ in batch_load_unet3d(preprocessed_dataset_dir, val=val):
        pbar.update(x.shape[0])

  def load_resnet(val):
    from extra.datasets.imagenet import get_train_files, get_val_files
    files = get_val_files() if val else get_train_files()
    with tqdm(total=len(files)) as pbar:
      for x,y,c in batch_load_resnet(val=val):
        pbar.update(x.shape[0])
  def load_retinanet(val):
    from extra.datasets.openimages import get_retinanet_train_files, get_retinanet_val_files
    from examples.mlperf.helpers import anchor_generator
    feature_shapes = [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]
    ANCHORS = anchor_generator((10,3,800,800), feature_shapes)
    ANCHOR_NP = ANCHORS[0].numpy()
    files = get_retinanet_val_files() if val else get_retinanet_train_files()
    cnt=0
    with tqdm(total=len(files)) as pbar:
      for x in batch_load_retinanet(96, val=val, anchor_np=ANCHOR_NP, pad_first_batch=False):
        pbar.update(x[0].shape[0])

  load_fn_name = f"load_{getenv('MODEL', 'resnet')}"
  if load_fn_name in globals():
    globals()[load_fn_name](getenv("VAL", 1))
