from tinygrad.tensor import Tensor
from tinygrad.features.jit import TinyJit
from tinygrad.nn import Linear, Embedding
from tinygrad.helpers import fetch
import numpy as np
from pathlib import Path

# https://assets.amazon.science/6e/5f/5ef4386f4e0d896d612284c9b9b6/efficient-minimum-word-error-rate-training-of-rnn-transducer-for-end-to-end-speech-recognition.pdf
# https://github.com/HawkAaron/RNN-Transducer/blob/graves2013/rnnt_np.py
def logsumexp(x1:Tensor, x2:Tensor) -> Tensor:
  return (x1.exp() + x2.exp()).log()
def forward_pass(log_probs, labels, blank):
  T, U, _ = log_probs.shape
  # alphas = np.zeros((T, U))
  alphas = Tensor.zeros((T, U))

  for t in range(1, T):
    alphas[t, 0] = alphas[t-1, 0] + log_probs[t-1, 0, blank]

  for u in range(1, U):
    alphas[0, u] = alphas[0, u-1] + log_probs[0, u-1, labels[u-1]]
  for t in range(1, T):
    for u in range(1, U):
      no_emit = alphas[t-1, u] + log_probs[t-1, u, blank]
      emit = alphas[t, u-1] + log_probs[t, u-1, labels[u-1]]
      # alphas[t, u] = np.logaddexp(emit, no_emit)
      alphas[t, u] = logsumexp(emit, no_emit)
          
          

  loglike = alphas[T-1, U-1] + log_probs[T-1, U-1, blank]
  return alphas, loglike

def backward_pass(log_probs, labels, blank):

  T, U, _ = log_probs.shape
  # betas = np.zeros((T, U))
  betas = Tensor.zeros((T, U))
  betas[T-1, U-1] = log_probs[T-1, U-1, blank]

  for t in reversed(range(T-1)):
    betas[t, U-1] = betas[t+1, U-1] + log_probs[t, U-1, blank]

  for u in reversed(range(U-1)):
    betas[T-1, u] = betas[T-1, u+1] + log_probs[T-1, u, labels[u]]

  for t in reversed(range(T-1)):
    for u in reversed(range(U-1)):
      no_emit = betas[t+1, u] + log_probs[t, u, blank]
      emit = betas[t, u+1] + log_probs[t, u, labels[u]]
      # betas[t, u] = np.logaddexp(emit, no_emit)
      betas[t, u] = logsumexp(emit, no_emit)

  return betas, betas[0, 0]

def compute_gradient(log_probs, alphas, betas, labels, blank):
  T, U, _ = log_probs.shape
  # grads = np.full(log_probs.shape, -float("inf"))
  grads = Tensor.full(log_probs.shape, -float("inf"))
  log_like = betas[0, 0]

  grads[T-1, U-1, blank] = alphas[T-1, U-1]

  grads[:T-1, :, blank] = alphas[:T-1, :] + betas[1:, :]
  for u, l in enumerate(labels):
    grads[:, u, l] = alphas[:, u] + betas[:, u+1]

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

def transduce_batch_helper(log_probs, labels, xlen, ylen, blank=0):
  # grads = np.zeros_like(log_probs)
  grads = Tensor.zeros_like(log_probs)
  costs = []

  for b in range(log_probs.shape[0]):
    t = int(xlen[b].item())
    u = int(ylen[b].item()) + 1
    ll, g = transduce(log_probs[b, :t, :u, :], labels[b, :u-1], blank)
    grads[b, :t, :u, :] = g
    costs.append(ll)
  return costs, grads





# class Loss:
#     def __init__(self, phi_idx: int) -> None:
#         # super().__init__()
#         self.phi_idx = phi_idx

#     def forward(
#             self,
#             probs: Tensor,
#             target: Tensor,
#             target_lengths: Tensor
#             ) -> Tensor:
#         # target_lengths = target_lengths.to(self.device)
#         batch_size, max_length, *_ = probs.shape
#         n_chars = target_lengths.max().item()
#         n_nulls = max_length - n_chars
#         # initializing the scores matrix
#         scores = self.get_score_matrix(batch_size, n_chars, n_nulls)
#         # scores = scores.to(self.device)
#         # going over all possible alignment paths
#         for c in range(n_chars + 1):
#             for p in range(n_nulls + 1):
#                 if c == 0 and p == 0:
#                     # keeping scores[:, c, p] zeros
#                     continue
#                 scores = self.update_scores(scores, probs, target, p, c)
#         return self.calc_loss(scores, target_lengths)

#     def calc_loss(self, scores: Tensor, target_lengths: Tensor) -> Tensor:
#         """Calculates the loss from the given loglikelhood of all paths

#         Args:
#             scores (Tensor): The score matrix
#             target_lengths (Tensor): A tensor contains the lengths of
#             the true target

#         Returns:
#             Tensor: The loss
#         """
#         # should we normalize by the number of paths ?
#         loss = torch.diagonal(torch.index_select(
#             scores[:, :, -1], dim=1, index=target_lengths
#             ))
#         # loss = 
#         loss = -1 * loss
#         return loss.mean()

#     def get_score_matrix(
#             self, batch_size: int, n_chars: int, n_nulls: int
#             ) -> Tensor:
#         """Returns a zeros matrix with (B, n_chars, n_nulls) shape

#         Args:
#             batch_size (int): the target batch size
#             n_chars (int): the number of maximum length of chars
#             n_nulls (int): the number of nulls to be added to reach the
#             maximum length

#         Returns:
#             Tensor: Zeros matrix with (B, n_chars, n_nulls) shape
#         """
#         # return torch.zeros(batch_size, n_chars + 1, n_nulls + 1)
#         return Tensor.zeros(batch_size, n_chars + 1, n_nulls + 1)

#     def update_scores(
#             self, scores: Tensor, probs: Tensor, target: Tensor, p: int, c: int
#             ) -> Tensor:
#         """Updates the given scores matrix based on the values of p and c

#         Args:
#             scores (Tensor): The scores matrix
#             probs (Tensor): The probabilities scores out of the model
#             target (Tensor): The target values
#             p (int): The location on the nulls dimension in the scores
#             matrix
#             c (int): The location on the characters dimension in the scores
#             matrix

#         Returns:
#             Tensor: The updated scores matrix
#         """
#         if p == 0:
#             chars_probs = self.get_chars_probs(probs, target, c, p)
#             scores[:, c, p] = chars_probs + scores[:, c - 1, p]
#             return scores
#         elif c == 0:
#             phi_probs = self.get_phi_probs(probs, c, p)
#             scores[:, c, p] = phi_probs + scores[:, c, p - 1]
#             return scores
#         chars_probs = self.get_chars_probs(probs, target, c, p)
#         phi_probs = self.get_phi_probs(probs, c, p)
#         scores[:, c, p] = torch.logsumexp(
#             torch.stack(
#                 [scores[:, c, p - 1] + self.log(phi_probs),
#                 scores[:, c - 1, p] + self.log(chars_probs)]
#             ), dim=0)
#         return scores

#     def get_phi_probs(self, probs: Tensor, c: int, p: int) -> Tensor:
#         return probs[:, c + p - 1, self.phi_idx]

#     def get_chars_probs(
#             self, probs: Tensor, target: Tensor, c: int, p: int
#             ) -> Tensor:
#         all_seqs = probs[:, p + c - 1]
#         result = torch.index_select(all_seqs, dim=-1, index=target[:, c - 1])
#         return torch.diagonal(result)


class RNNT:
  def __init__(self, input_features=240, vocab_size=29, enc_hidden_size=1024, pred_hidden_size=320, joint_hidden_size=512, pre_enc_layers=2, post_enc_layers=3, pred_layers=2, stack_time_factor=2, dropout=0.32):
    self.encoder = Encoder(input_features, enc_hidden_size, pre_enc_layers, post_enc_layers, stack_time_factor, dropout)
    self.prediction = Prediction(vocab_size, pred_hidden_size, pred_layers, dropout)
    self.joint = Joint(vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout)

  @TinyJit
  def __call__(self, x, y, hc=None):
    f, _ = self.encoder(x, None)
    g, _ = self.prediction(y, hc, Tensor.ones(1, requires_grad=False))
    out = self.joint(f, g)
    return out.realize()

  def decode(self, x, x_lens):
    logits, logit_lens = self.encoder(x, x_lens)
    outputs = []
    for b in range(logits.shape[0]):
      inseq = logits[b, :, :].unsqueeze(1)
      logit_len = logit_lens[b]
      seq = self._greedy_decode(inseq, int(np.ceil(logit_len.numpy()).item()))
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

  @TinyJit
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
