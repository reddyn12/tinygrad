from tinygrad.tensor import Tensor
from tinygrad.nn import optim, state
from tinygrad.helpers import getenv

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  from extra.models import rnnt
  from extra.datasets.librispeech import tokenize_transcripts, iterate_new,iterate 
  from examples.mlperf.metrics import word_error_rate
  config = {}
  seed = config["seed"] = getenv("SEED", 42)
  Tensor.manual_seed(seed)  # seed for weight initialization
  
  model = rnnt.RNNT()
  model.load_from_pretrained()
  # loss_fn = rnnt.RNNT_LOSS
  loss_fn = rnnt.transduce_batch_helper
  tokenizer = rnnt.Tokenizer()
  optimizer = optim.LAMB(state.get_parameters(model))
  
  # X, Y = next(iterate(bs=4))
  # x = Tensor(X[0])
  # x_lens = Tensor([X[1]])
  # y = tokenizer.tokenize(Y[0])
  # y = tokenize_transcripts(Y.tolist())  
  # y_tensors = [Tensor(t) for t in y[0]]
  # y_lens = Tensor(y[1])
  
  x, y, x_len, y_len = next(iterate_new(bs=4))
  print(x.shape, y.shape, x_len.shape, y_len.shape)
  output = model(x, y).log_softmax()
  output = output.realize()
  # loss, grad = loss_fn(output)
  losses, output.grad = loss_fn(output, y, x_len, y_len)
  print(output.shape, output.grad, output.grad.shape, losses)
  
  

  
  
  
  
  pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


