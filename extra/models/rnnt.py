from tinygrad.tensor import Tensor, Function
from tinygrad.features.jit import TinyJit
from tinygrad.nn import Linear, Embedding
from tinygrad.helpers import fetch
import numpy as np
from pathlib import Path
# MLPERF Config: https://github.com/mlcommons/training/blob/master/rnn_speech_recognition/pytorch/configs/baseline_v3-1023sp.yaml
# tokenizer:
#   sentpiece_model: /datasets/sentencepieces/librispeech1023.model
#   labels: [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
#            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
             "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
BLANK = len(LABELS)
class Tokenizer:
  def __init__(self, labels=LABELS, sentpiece_model=None):
    """Converts transcript to a sequence of tokens.

    Args:
        labels (str): all possible output symbols
    """
    # For labels use vocab or load worpieces
    self.charset = labels
    self.use_sentpiece = (sentpiece_model is not None)
    if self.use_sentpiece:
      import sentencepiece as spm
      self.sentpiece = spm.SentencePieceProcessor(model_file=sentpiece_model)
      self.num_labels = len(self.sentpiece)
    else:
      self.num_labels = len(self.charset)
      self.label2ind = {lab: i for i, lab in enumerate(self.charset)}

  def tokenize(self, transcript):
    if self.use_sentpiece:
      inds = self.sentpiece.encode(transcript, out_type=int)
      assert 0 not in inds, '<unk> found during tokenization (OOV?)'
    else:
      inds = [self.label2ind[x]
      for x in transcript if x in self.label2ind]
    return inds

  def detokenize(self, inds):
    if self.use_sentpiece:
      return self.sentpiece.decode(inds)
    else:
      return ''.join(self.charset[i] for i in inds)

# https://assets.amazon.science/6e/5f/5ef4386f4e0d896d612284c9b9b6/efficient-minimum-word-error-rate-training-of-rnn-transducer-for-end-to-end-speech-recognition.pdf
# https://github.com/HawkAaron/RNN-Transducer/blob/graves2013/rnnt_np.py
# @TinyJit
def logsumexp(x1:Tensor, x2:Tensor) -> Tensor:
  # return (x1.exp() + x2.exp()).log()
  temp = (x1.exp() + x2.exp()).log()
  temp.requires_grad = False
  return temp
# @TinyJit
def forward_pass(log_probs, labels, blank=BLANK):
  T, U, _ = log_probs.shape
  # alphas = np.zeros((T, U))
  alphas = Tensor.zeros((T, U), requires_grad=False)

  for t in range(1, T):
    temp = alphas[t-1, 0] + log_probs[t-1, 0, blank]
    temp.requires_grad = False
    
    alphas[t, 0] = temp

  for u in range(1, U):
    temp = alphas[0, u-1] + log_probs[0, u-1, int(labels[u-1].item())]
    temp.requires_grad = False
    alphas[0, u] = temp
  for t in range(1, T):
    for u in range(1, U):
      no_emit = alphas[t-1, u] + log_probs[t-1, u, blank]
      emit = alphas[t, u-1] + log_probs[t, u-1, int(labels[u-1].item())]
      # alphas[t, u] = np.logaddexp(emit, no_emit)
      alphas[t, u] = logsumexp(emit, no_emit)
          
          

  loglike = alphas[T-1, U-1] + log_probs[T-1, U-1, blank]
  return alphas, loglike

def backward_pass(log_probs, labels, blank=BLANK):

  T, U, _ = log_probs.shape
  # betas = np.zeros((T, U))
  betas = Tensor.zeros((T, U), requires_grad=False)
  temp = log_probs[T-1, U-1, blank]
  temp.requires_grad = False
  betas[T-1, U-1] = temp

  for t in reversed(range(T-1)):
    temp = betas[t+1, U-1] + log_probs[t, U-1, blank]
    temp.requires_grad = False
    betas[t, U-1] = temp

  for u in reversed(range(U-1)):
    temp = betas[T-1, u+1] + log_probs[T-1, u, int(labels[u].item())]
    temp.requires_grad = False
    betas[T-1, u] = temp

  for t in reversed(range(T-1)):
    for u in reversed(range(U-1)):
      no_emit = betas[t+1, u] + log_probs[t, u, blank]
      emit = betas[t, u+1] + log_probs[t, u, int(labels[u].item())]
      # betas[t, u] = np.logaddexp(emit, no_emit)
      betas[t, u] = logsumexp(emit, no_emit)

  return betas, betas[0, 0]

def compute_gradient(log_probs, alphas, betas, labels, blank):
  T, U, _ = log_probs.shape
  # grads = np.full(log_probs.shape, -float("inf"))
  grads = Tensor.full(log_probs.shape, -float("inf"))
  # grads.requires_grad = False
  log_like = betas[0, 0]

  grads[T-1, U-1, blank] = alphas[T-1, U-1]

  grads[:T-1, :, blank] = alphas[:T-1, :] + betas[1:, :]
  
  
  for u, l in enumerate(labels):
    # print('COMPUTE GRAD',labels.shape, u, u+1)
    # temp hack?
    temp = alphas[:, u//labels.shape[0]] + betas[:, (u+1)//labels.shape[0]]
    temp.requires_grad = False
    grads[:, u, int(l.item())] = temp

  # grads = -np.exp(grads + log_probs - log_like)
  grads = -((grads + log_probs - log_like).exp())
  return grads

def transduce(log_probs, labels, blank=0):
  """
  Args:
      log_probs: 3D array with shape
            [input len, output len + 1, vocab size]
      labels: 1D array with shape [output time steps]
  Returns:
      float: The negative log-likelihood
      3D array: Gradients with respect to the
                  unnormalized input actications
  """
  alphas, ll_forward = forward_pass(log_probs, labels, blank)
  betas, ll_backward = backward_pass(log_probs, labels, blank)
  grads = compute_gradient(log_probs, alphas, betas, labels, blank)
  return -ll_forward, grads

def transduce_batch_helper(log_probs, labels, xlen, ylen, blank=BLANK):
  # grads = np.zeros_like(log_probs)
  grads = Tensor.zeros_like(log_probs)
  # grads.requires_grad = False
  # costs = Tensor.empty((log_probs.shape[0],))
  costs = []

  for b in range(log_probs.shape[0]):
    t = int(xlen[b].item())
    u = int(ylen[b].item()) + 1
    ll, g = transduce(log_probs[b, :t, :u, :], labels[b, :u-1], blank)
    # g.requires_grad = False
    grads[b, :t, :u, :] = g.numpy()
    # ll.requires_grad = False
    # costs[b] = ll
    costs.append(ll)
  grads.realize()
  return costs, grads
class RNNT_LOSS(Function):
  # @staticmethod
  def forward(self, log_probs, labels):
    print('LOSS_Forward:', log_probs.shape, labels.shape)
    B, T, C, _ = log_probs.shape
    # a = Tensor.empty((B,T,C), requires_grad=True)
    
    # l = Tensor.empty((B,), requires_grad=True)
    a = [None]*B
    l = [None]*B
    for i in range(B):
      
      at,lt = forward_pass(log_probs[i], labels[i])
      # at.realize()
      # lt.realize()
      # at.requires_grad=False
      # lt.requires_grad=False
      a[i], l[i] = at, lt
   
    a = Tensor.stack(a)
    l = Tensor.stack(l)
    # a,l = forward_pass(log_probs[0], labels[0])
    
    print('AAAAA:', a.shape)
    
    
    return -l
  # @staticmethod
  def backward(self, log_probs, labels):
    b,l = backward_pass(log_probs, labels)
    return l
  def __call__(self, log_probs, labels):
    return self.forward(log_probs, labels)

def train_epoch():
  pass
# def zero_pad_concat(inputs):
#   max_t = max(inp.shape[0] for inp in inputs)
#   shape = (len(inputs), max_t) + inputs[0].shape[1:]
#   input_mat = np.zeros(shape, dtype=np.float32)
#   for e, inp in enumerate(inputs):
#     input_mat[e, :inp.shape[0]] = inp
#   return input_mat

# def end_pad_concat(inputs):
#   max_t = max(i.shape[0] for i in inputs)
#   shape = (len(inputs), max_t)
#   labels = np.full(shape, fill_value=inputs[0][-1], dtype='i')
#   for e, l in enumerate(inputs):
#     labels[e, :len(l)] = l
#   return labels

# def convert(inputs, labels, ctx):
#   # length no need move to gpu
#   xlen = mx.nd.array([i.shape[0] for i in inputs], ctx=ctx)
#   ylen = mx.nd.array([i.shape[0] for i in labels], ctx=ctx)
#   xs = mx.nd.array(zero_pad_concat(inputs), ctx=ctx)
#   ys = mx.nd.array(end_pad_concat(labels), ctx=ctx)
#   return xs, ys, xlen, ylen
class RNNT:
  def __init__(self, input_features=240, vocab_size=29, enc_hidden_size=1024, pred_hidden_size=320, joint_hidden_size=512, pre_enc_layers=2, post_enc_layers=3, pred_layers=2, stack_time_factor=2, dropout=0.32):
    self.encoder = Encoder(input_features, enc_hidden_size, pre_enc_layers, post_enc_layers, stack_time_factor, dropout)
    self.prediction = Prediction(vocab_size, pred_hidden_size, pred_layers, dropout)
    self.joint = Joint(vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout)

  @TinyJit
  def __call__(self, x, y, hc=None):
    f, _ = self.encoder(x, None)
    g, _ = self.prediction(y, hc, Tensor.ones(1, requires_grad=False))
    # g, _ = self.prediction(y, hc, Tensor.ones(1))

    out = self.joint(f, g)
    return out.realize()

  def decode(self, x, x_lens):
    logits, logit_lens = self.encoder(x, x_lens)
    outputs = []
    for b in range(logits.shape[0]):
      inseq = logits[b, :, :].unsqueeze(1)
      logit_len = logit_lens[b]
      # seq = self._greedy_decode(inseq, int(np.ceil(logit_len.numpy()).item()))
      seq = self._greedy_decode(inseq, int(Tensor.ceil(logit_len).item()))
      outputs.append(seq)
    return outputs

  def _greedy_decode(self, logits, logit_len):
    hc = Tensor.zeros(self.prediction.rnn.layers, 2, self.prediction.hidden_size, requires_grad=False)
    labels = []
    label = Tensor.zeros(1, 1, requires_grad=False)
    mask = Tensor.zeros(1, requires_grad=False)
    for time_idx in range(logit_len):
      logit = logits[time_idx, :, :].unsqueeze(0)
      not_blank = True
      added = 0
      while not_blank and added < 30:
        if len(labels) > 0:
          mask = (mask + 1).clip(0, 1)
          label = Tensor([[labels[-1] if labels[-1] <= 28 else labels[-1] - 1]], requires_grad=False) + 1 - 1
        jhc = self._pred_joint(Tensor(logit.numpy()), label, hc, mask)
        k = jhc[0, 0, :29].argmax(axis=0).numpy()
        not_blank = k != 28
        if not_blank:
          labels.append(k)
          hc = jhc[:, :, 29:] + 1 - 1
        added += 1
    return labels

  # @TinyJit
  def _pred_joint(self, logit, label, hc, mask):
    g, hc = self.prediction(label, hc, mask)
    j = self.joint(logit, g)[0]
    j = j.pad(((0, 1), (0, 1), (0, 0)))
    out = j.cat(hc, dim=2)
    return out.realize()

  def load_from_pretrained(self):
    fn = Path(__file__).parents[1] / "weights/rnnt.pt"
    fetch("https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1", fn)

    import torch
    with open(fn, "rb") as f:
      state_dict = torch.load(f, map_location="cpu")["state_dict"]

    # encoder
    for i in range(2):
      self.encoder.pre_rnn.cells[i].weights_ih.assign(state_dict[f"encoder.pre_rnn.lstm.weight_ih_l{i}"].numpy())
      self.encoder.pre_rnn.cells[i].weights_hh.assign(state_dict[f"encoder.pre_rnn.lstm.weight_hh_l{i}"].numpy())
      self.encoder.pre_rnn.cells[i].bias_ih.assign(state_dict[f"encoder.pre_rnn.lstm.bias_ih_l{i}"].numpy())
      self.encoder.pre_rnn.cells[i].bias_hh.assign(state_dict[f"encoder.pre_rnn.lstm.bias_hh_l{i}"].numpy())
    for i in range(3):
      self.encoder.post_rnn.cells[i].weights_ih.assign(state_dict[f"encoder.post_rnn.lstm.weight_ih_l{i}"].numpy())
      self.encoder.post_rnn.cells[i].weights_hh.assign(state_dict[f"encoder.post_rnn.lstm.weight_hh_l{i}"].numpy())
      self.encoder.post_rnn.cells[i].bias_ih.assign(state_dict[f"encoder.post_rnn.lstm.bias_ih_l{i}"].numpy())
      self.encoder.post_rnn.cells[i].bias_hh.assign(state_dict[f"encoder.post_rnn.lstm.bias_hh_l{i}"].numpy())

    # prediction
    self.prediction.emb.weight.assign(state_dict["prediction.embed.weight"].numpy())
    for i in range(2):
      self.prediction.rnn.cells[i].weights_ih.assign(state_dict[f"prediction.dec_rnn.lstm.weight_ih_l{i}"].numpy())
      self.prediction.rnn.cells[i].weights_hh.assign(state_dict[f"prediction.dec_rnn.lstm.weight_hh_l{i}"].numpy())
      self.prediction.rnn.cells[i].bias_ih.assign(state_dict[f"prediction.dec_rnn.lstm.bias_ih_l{i}"].numpy())
      self.prediction.rnn.cells[i].bias_hh.assign(state_dict[f"prediction.dec_rnn.lstm.bias_hh_l{i}"].numpy())

    # joint
    self.joint.l1.weight.assign(state_dict["joint_net.0.weight"].numpy())
    self.joint.l1.bias.assign(state_dict["joint_net.0.bias"].numpy())
    self.joint.l2.weight.assign(state_dict["joint_net.3.weight"].numpy())
    self.joint.l2.bias.assign(state_dict["joint_net.3.bias"].numpy())


class LSTMCell:
  def __init__(self, input_size, hidden_size, dropout):
    self.dropout = dropout

    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
    self.bias_ih = Tensor.uniform(hidden_size * 4)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
    self.bias_hh = Tensor.uniform(hidden_size * 4)

  def __call__(self, x, hc):
    gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)

    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

    c = (f * hc[x.shape[0]:]) + (i * g)
    h = (o * c.tanh()).dropout(self.dropout)

    return Tensor.cat(h, c).realize()


class LSTM:
  def __init__(self, input_size, hidden_size, layers, dropout):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layers = layers

    self.cells = [LSTMCell(input_size, hidden_size, dropout) if i == 0 else LSTMCell(hidden_size, hidden_size, dropout if i != layers - 1 else 0) for i in range(layers)]

  def __call__(self, x, hc):
    @TinyJit
    def _do_step(x_, hc_):
      return self.do_step(x_, hc_)

    if hc is None:
      hc = Tensor.zeros(self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False)

    output = None
    for t in range(x.shape[0]):
      hc = _do_step(x[t] + 1 - 1, hc) # TODO: why do we need to do this?
      if output is None:
        output = hc[-1:, :x.shape[1]]
      else:
        output = output.cat(hc[-1:, :x.shape[1]], dim=0).realize()

    return output, hc

  def do_step(self, x, hc):
    new_hc = [x]
    for i, cell in enumerate(self.cells):
      new_hc.append(cell(new_hc[i][:x.shape[0]], hc[i]))
    return Tensor.stack(new_hc[1:]).realize()


class StackTime:
  def __init__(self, factor):
    self.factor = factor

  def __call__(self, x, x_lens):
    x = x.pad(((0, (-x.shape[0]) % self.factor), (0, 0), (0, 0)))
    x = x.reshape(x.shape[0] // self.factor, x.shape[1], x.shape[2] * self.factor)
    return x, x_lens / self.factor if x_lens is not None else None


class Encoder:
  def __init__(self, input_size, hidden_size, pre_layers, post_layers, stack_time_factor, dropout):
    self.pre_rnn = LSTM(input_size, hidden_size, pre_layers, dropout)
    self.stack_time = StackTime(stack_time_factor)
    self.post_rnn = LSTM(stack_time_factor * hidden_size, hidden_size, post_layers, dropout)

  def __call__(self, x, x_lens):
    x, _ = self.pre_rnn(x, None)
    x, x_lens = self.stack_time(x, x_lens)
    x, _ = self.post_rnn(x, None)
    return x.transpose(0, 1), x_lens


class Prediction:
  def __init__(self, vocab_size, hidden_size, layers, dropout):
    self.hidden_size = hidden_size

    self.emb = Embedding(vocab_size - 1, hidden_size)
    self.rnn = LSTM(hidden_size, hidden_size, layers, dropout)

  def __call__(self, x, hc, m):
    emb = self.emb(x) * m
    x_, hc = self.rnn(emb.transpose(0, 1), hc)
    return x_.transpose(0, 1), hc


class Joint:
  def __init__(self, vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout):
    self.dropout = dropout

    self.l1 = Linear(pred_hidden_size + enc_hidden_size, joint_hidden_size)
    self.l2 = Linear(joint_hidden_size, vocab_size)

  def __call__(self, f, g):
    (_, T, H), (B, U, H2) = f.shape, g.shape
    f = f.unsqueeze(2).expand(B, T, U, H)
    g = g.unsqueeze(1).expand(B, T, U, H2)

    inp = f.cat(g, dim=3)
    t = self.l1(inp).relu()
    t = t.dropout(self.dropout)
    return self.l2(t)
