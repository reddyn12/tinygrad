import math, sys
from tinygrad import Tensor, dtypes
from tinygrad.helpers import flatten, get_child
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.nn import Conv2d
from extra.models.resnet import ResNet
from examples.mlperf.losses import sigmoid_focal_loss, l1_loss
from examples.mlperf.helpers import encode_boxes
import numpy as np
import line_profiler
import torch, torchvision
from typing import List, Dict, Tuple

TupleOf36Pairs = Tuple[
    Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int],
    Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int],
    Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int],
    Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int],
    Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int],
    Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]
]

Conv2dNormal = Conv2d
Conv2dNormal_priorprob = Conv2d
Conv2dKaiming = Conv2d

@line_profiler.profile
def nms(boxes, scores, thresh=0.5):
  x1, y1, x2, y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  to_process, keep = scores.argsort()[::-1], []
  while to_process.size > 0:
    cur, to_process = to_process[0], to_process[1:]
    keep.append(cur)
    inter_x1 = np.maximum(x1[cur], x1[to_process])
    inter_y1 = np.maximum(y1[cur], y1[to_process])
    inter_x2 = np.minimum(x2[cur], x2[to_process])
    inter_y2 = np.minimum(y2[cur], y2[to_process])
    a1a = inter_x2 - inter_x1 + 1
    a2a = inter_y2 - inter_y1 + 1
    a1 = np.maximum(0, a1a)
    a2 = np.maximum(0, a2a)
    inter_area = a1 * a2
    iou = inter_area / (areas[cur] + areas[to_process] - inter_area)
    to_process = to_process[np.where(iou <= thresh)[0]]
  return keep

def nms_torch(boxes:torch.Tensor, scores:torch.Tensor, thresh:float =0.5):
  return torch.ops.torchvision.nms(boxes.float(), scores.float(), thresh)
  x1, y1, x2, y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  to_process, keep = scores.argsort().flip((0,)), []
  while to_process.numel() > 0:
    cur, to_process = to_process[0], to_process[1:]
    keep.append(cur)
    inter_x1 = torch.maximum(x1[cur], x1[to_process])
    inter_y1 = torch.maximum(y1[cur], y1[to_process])
    inter_x2 = torch.minimum(x2[cur], x2[to_process])
    inter_y2 = torch.minimum(y2[cur], y2[to_process])
    a1a = inter_x2 - inter_x1 + 1
    a2a = inter_y2 - inter_y1 + 1
    a1 = torch.maximum(0, a1a)
    a2 = torch.maximum(0, a2a)
    inter_area = a1 * a2
    iou = inter_area / (areas[cur] + areas[to_process] - inter_area)
    to_process = to_process[torch.where(iou <= thresh)[0]]
  return keep

def decode_bbox(offsets, anchors):
  dx, dy, dw, dh = offsets[:,0],offsets[:,1],offsets[:,2],offsets[:,3]
  widths, heights = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
  cx, cy = anchors[:, 0] + 0.5 * widths, anchors[:, 1] + 0.5 * heights
  pred_cx, pred_cy = dx * widths + cx, dy * heights + cy
  pred_w, pred_h = np.exp(dw) * widths, np.exp(dh) * heights
  pred_x1, pred_y1 = pred_cx - 0.5 * pred_w, pred_cy - 0.5 * pred_h
  pred_x2, pred_y2 = pred_cx + 0.5 * pred_w, pred_cy + 0.5 * pred_h
  return np.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=1, dtype=np.float32)
def decode_bbox_torch(rel_codes, boxes, bbox_clip: float =math.log(1000. / 16)):
  """
  From a set of original boxes and encoded relative box offsets,
  get the decoded boxes.

  Args:
      rel_codes (Tensor): encoded boxes
      boxes (Tensor): reference boxes.
  """

  boxes = boxes.to(rel_codes.dtype)

  widths = boxes[:, 2] - boxes[:, 0]
  heights = boxes[:, 3] - boxes[:, 1]
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  # wx, wy, ww, wh = self.weights
  dx = rel_codes[:, 0::4] #/ wx
  dy = rel_codes[:, 1::4] #/ wy
  dw = rel_codes[:, 2::4]# / ww
  dh = rel_codes[:, 3::4]#/ wh

  # Prevent sending too large values into torch.exp()
  dw = torch.clamp(dw, max=bbox_clip)
  dh = torch.clamp(dh, max=bbox_clip)

  pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
  pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
  pred_w = torch.exp(dw) * widths[:, None]
  pred_h = torch.exp(dh) * heights[:, None]

  # Distance from center to box's corner.
  c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
  c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

  pred_boxes1 = pred_ctr_x - c_to_c_w
  pred_boxes2 = pred_ctr_y - c_to_c_h
  pred_boxes3 = pred_ctr_x + c_to_c_w
  pred_boxes4 = pred_ctr_y + c_to_c_h
  pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
  return pred_boxes

def generate_anchors(input_size, grid_sizes, scales, aspect_ratios):
  assert len(scales) == len(aspect_ratios) == len(grid_sizes)
  anchors = []
  for s, ar, gs in zip(scales, aspect_ratios, grid_sizes):
    s, ar = np.array(s), np.array(ar)
    h_ratios = np.sqrt(ar)
    w_ratios = 1 / h_ratios
    ws = (w_ratios[:, None] * s[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * s[None, :]).reshape(-1)
    base_anchors = (np.stack([-ws, -hs, ws, hs], axis=1) / 2).round()
    stride_h, stride_w = input_size[0] // gs[0], input_size[1] // gs[1]
    shifts_x, shifts_y = np.meshgrid(np.arange(gs[1]) * stride_w, np.arange(gs[0]) * stride_h)
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    shifts = np.stack([shifts_x, shifts_y, shifts_x, shifts_y], axis=1, dtype=np.float32)
    anchors.append(torch.from_numpy((shifts[:, None] + base_anchors[None, :]).reshape(-1, 4)).to(torch.float16))
  return anchors

class RetinaNet:
  def __init__(self, backbone: ResNet, num_classes=264, num_anchors=9, scales=None, aspect_ratios=None):
    assert isinstance(backbone, ResNet)
    scales = tuple((i, int(i*2**(1/3)), int(i*2**(2/3))) for i in [32,  64, 128, 256, 512]) if scales is None else scales
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(scales) if aspect_ratios is None else aspect_ratios
    self.num_anchors, self.num_classes = num_anchors, num_classes
    assert len(scales) == len(aspect_ratios) and all(self.num_anchors == len(s) * len(ar) for s, ar in zip(scales, aspect_ratios))

    self.backbone = ResNetFPN(backbone)
    self.head = RetinaHead(self.backbone.out_channels, num_anchors=num_anchors, num_classes=num_classes)
    self.anchor_gen = lambda input_size: generate_anchors(input_size, self.backbone.compute_grid_sizes(input_size), scales, aspect_ratios)
    self.anchors = self.anchor_gen((800,800))

  def __call__(self, x, split=False):
    return self.forward(x, split)

  def forward(self, x, split):
    b = self.backbone(x)
    r, c = self.head(b)
    if split: return b, r, c
    else: return r.cat(c.sigmoid(), dim=-1)

  def load_from_pretrained(self):
    model_urls = {
      (50, 1, 64): "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
      (50, 32, 4): "https://zenodo.org/record/6605272/files/retinanet_model_10.zip",
    }
    self.url = model_urls[(self.backbone.body.num, self.backbone.body.groups, self.backbone.body.base_width)]
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url(self.url, progress=True, map_location='cpu')
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    for k, v in state_dict.items():
      obj = get_child(self, k)
      dat = v.detach().numpy()
      assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat)

  def load_checkpoint(self, fn):
    d = safe_load(fn)
    load_state_dict(self, d)

  # predictions: (BS, (H1W1+...+HmWm)A, 4 + K)
  @line_profiler.profile
  def postprocess_detections(self, offsets, scores, input_size=(800, 800), image_sizes=None, orig_image_sizes=None, score_thresh=0.05, topk_candidates=1000, nms_thresh=0.5):
    anchors = self.anchor_gen(input_size)
    detections = []
    for i in range(offsets[0].shape[0]):
      h, w = input_size if image_sizes is None else image_sizes[i]

      offsets_per_image = [br[i] for br in offsets]
      scores_per_image = [cl[i] for cl in scores]

      image_boxes, image_scores, image_labels = [], [], []
      for offsets_per_level, scores_per_level, anchors_per_level in zip(offsets_per_image, scores_per_image, anchors):
        # remove low scoring boxes
        topk_idxs = np.where(scores_per_level > score_thresh)[0]
        scores_per_level = scores_per_level[topk_idxs]

        # keep topk
        num_topk = min(len(topk_idxs), topk_candidates)
        sort_idxs = scores_per_level.argsort()[-num_topk:][::-1]
        topk_idxs, scores_per_level = topk_idxs[sort_idxs], scores_per_level[sort_idxs]

        # bbox coords from offsets
        anchor_idxs = topk_idxs // self.num_classes
        labels_per_level = topk_idxs % self.num_classes
        boxes_per_level = decode_bbox(offsets_per_level[anchor_idxs], anchors_per_level[anchor_idxs])
        # clip to image size
        clipped_x = boxes_per_level[:, 0::2].clip(0, w)
        clipped_y = boxes_per_level[:, 1::2].clip(0, h)
        boxes_per_level = np.stack([clipped_x, clipped_y], axis=2).reshape(-1, 4)

        image_boxes.append(boxes_per_level)
        image_scores.append(scores_per_level)
        image_labels.append(labels_per_level)

      image_boxes = np.concatenate(image_boxes)
      image_scores = np.concatenate(image_scores)
      image_labels = np.concatenate(image_labels)

      # nms for each class
      keep_mask = np.zeros_like(image_scores, dtype=bool)
      for class_id in np.unique(image_labels):
        curr_indices = np.where(image_labels == class_id)[0]
        curr_keep_indices = nms(image_boxes[curr_indices], image_scores[curr_indices], nms_thresh)
        keep_mask[curr_indices[curr_keep_indices]] = True
      keep = np.where(keep_mask)[0]
      keep = keep[image_scores[keep].argsort()[::-1]]

      # resize bboxes back to original size
      image_boxes = image_boxes[keep]
      if orig_image_sizes is not None:
        resized_x = image_boxes[:, 0::2] * orig_image_sizes[i][1] / w
        resized_y = image_boxes[:, 1::2] * orig_image_sizes[i][0] / h
        image_boxes = np.stack([resized_x, resized_y], axis=2).reshape(-1, 4)
      # xywh format
      image_boxes = np.concatenate([image_boxes[:, :2], image_boxes[:, 2:] - image_boxes[:, :2]], axis=1)

      detections.append({"boxes":image_boxes, "scores":image_scores[keep], "labels":image_labels[keep]})
    return detections
  # predictions: (BS, (H1W1+...+HmWm)A, 4 + K)
# @torch.jit.script
def postprocess_detections_torch(predictions, orig_image_sizes:TupleOf36Pairs=tuple([(100,100)]*36), score_thresh:float=0.05, topk_candidates:int=1000, nms_thresh:float =0.5, anchors=None):
  # anchors = self.anchor_gen(input_size)
  # grid_sizes = self.backbone.compute_grid_sizes(input_size)
  # split_idx = np.cumsum([int(self.num_anchors * sz[0] * sz[1]) for sz in grid_sizes[:-1]])
  split_idx = [90000, 22500, 5625, 1521, 441]
  detections:List[Dict[str, torch.Tensor]] = []
  for i, predictions_per_image in enumerate(predictions):
    # h, w = input_size if image_sizes is None else image_sizes[i]
    h, w = 800, 800

    # predictions_per_image = np.split(predictions_per_image, split_idx)
    predictions_per_image = torch.split(predictions_per_image, split_idx)
    offsets_per_image = [br[:, :4] for br in predictions_per_image]
    scores_per_image = [cl[:, 4:] for cl in predictions_per_image]

    image_boxes, image_scores, image_labels = [], [], []
    for offsets_per_level, scores_per_level, anchors_per_level in zip(offsets_per_image, scores_per_image, anchors):
      # remove low scoring boxes
      scores_per_level = scores_per_level.flatten()
      keep_idxs = scores_per_level > score_thresh
      scores_per_level = scores_per_level[keep_idxs]

      # keep topk
      topk_idxs = torch.where(keep_idxs)[0]
      num_topk = min(len(topk_idxs), topk_candidates)
      sort_idxs = scores_per_level.argsort()[-num_topk:].flip((0,))
      topk_idxs, scores_per_level = topk_idxs[sort_idxs], scores_per_level[sort_idxs]

      # bbox coords from offsets
      anchor_idxs = topk_idxs // 264
      labels_per_level = topk_idxs % 264
      boxes_per_level = decode_bbox_torch(offsets_per_level[anchor_idxs], anchors_per_level[anchor_idxs])
      # clip to image size
      clipped_x = boxes_per_level[:, 0::2].clip(0, w)
      clipped_y = boxes_per_level[:, 1::2].clip(0, h)
      boxes_per_level = torch.stack([clipped_x, clipped_y], dim=2).reshape(-1, 4)

      image_boxes.append(boxes_per_level)
      image_scores.append(scores_per_level)
      image_labels.append(labels_per_level)

    image_boxes = torch.concatenate(image_boxes)
    image_scores = torch.concatenate(image_scores)
    image_labels = torch.concatenate(image_labels)

    # nms for each class
    keep_mask = torch.zeros_like(image_scores, dtype=torch.bool)
    for class_id in torch.unique(image_labels):
      curr_indices = torch.where(image_labels == class_id)[0]
      curr_keep_indices = nms_torch(image_boxes[curr_indices], image_scores[curr_indices], nms_thresh)
      keep_mask[curr_indices[curr_keep_indices]] = True
    keep = torch.where(keep_mask)[0]
    keep = keep[image_scores[keep].argsort().flip((0,))]

    # resize bboxes back to original size
    image_boxes = image_boxes[keep]
    if orig_image_sizes is not None:
      resized_x = image_boxes[:, 0::2].to(torch.float32) * orig_image_sizes[i][1] / w
      resized_y = image_boxes[:, 1::2].to(torch.float32) * orig_image_sizes[i][0] / h
      image_boxes = torch.stack([resized_x, resized_y], dim=2).reshape(-1, 4)
    # xywh format
    image_boxes = torch.concatenate([image_boxes[:, :2], image_boxes[:, 2:] - image_boxes[:, :2]], dim=1)

    detections.append({"boxes":image_boxes.numpy(), "scores":image_scores[keep].numpy(), "labels":image_labels[keep].numpy()})
  return detections
class ClassificationHead:
  def __init__(self, in_channels, num_anchors, num_classes):
    self.num_classes = num_classes
    self.conv = flatten([(Conv2dNormal(in_channels, in_channels, kernel_size=3, padding=1), lambda x: x.relu()) for _ in range(4)])
    self.cls_logits = Conv2dNormal_priorprob(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

  def __call__(self, x):
    out = [self.cls_logits(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, self.num_classes) for feat in x]
    return out[0].cat(*out[1:], dim=1)

  def loss(self, logits_class, matched_idxs, labels):
    batch_size = logits_class.shape[0]
    foreground_idxs = matched_idxs >= 0
    num_foreground = foreground_idxs.sum(-1)

    labels = (labels + 1) * foreground_idxs - 1
    gt_classes_target = labels.one_hot(logits_class.shape[-1])
    valid_idxs = matched_idxs != -2
    return (sigmoid_focal_loss(logits_class, gt_classes_target, valid_idxs.reshape(batch_size, -1, 1)) / num_foreground).mean()

class RegressionHead:
  def __init__(self, in_channels, num_anchors):
    self.conv = flatten([(Conv2dNormal(in_channels, in_channels, kernel_size=3, padding=1), lambda x: x.relu()) for _ in range(4)])
    self.bbox_reg = Conv2dNormal(in_channels, num_anchors * 4, kernel_size=3, padding=1)

  def __call__(self, x):
    out = [self.bbox_reg(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, 4) for feat in x]
    return out[0].cat(*out[1:], dim=1)

  def loss(self, logits_reg, anchors, matched_idxs, boxes):
    foreground_idxs = matched_idxs >= 0
    num_foreground = foreground_idxs.sum(-1)
    mask = foreground_idxs.reshape(logits_reg.shape[0], -1, 1)

    matched_gt_boxes = boxes * mask
    bbox_reg = logits_reg * mask
    anchors = anchors * mask
    target_regression = encode_boxes(matched_gt_boxes, anchors)
    target_regression = target_regression * mask
    target_regression = (target_regression>-1000).where(target_regression, 0) #nan replace with 0
    return (l1_loss(bbox_reg, target_regression) / num_foreground).mean()

class RetinaHead:
  def __init__(self, in_channels, num_anchors, num_classes):
    self.classification_head = ClassificationHead(in_channels, num_anchors, num_classes)
    self.regression_head = RegressionHead(in_channels, num_anchors)

  def __call__(self, x):
    pred_bbox, pred_class = self.regression_head(x), self.classification_head(x)
    return pred_bbox, pred_class

class ResNetFPN:
  def __init__(self, resnet, out_channels=256, returned_layers=[2, 3, 4]):
    self.out_channels = out_channels
    self.body = resnet
    in_channels_list = [(self.body.in_planes // 8) * 2 ** (i - 1) for i in returned_layers]
    self.fpn = FPN(in_channels_list, out_channels)

  # this is needed to decouple inference from postprocessing (anchors generation)
  def compute_grid_sizes(self, input_size):
    return np.ceil(np.array(input_size)[None, :] / 2 ** np.arange(3, 8)[:, None])

  def __call__(self, x):
    out = self.body.bn1(self.body.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.body.layer1)
    p3 = out.sequential(self.body.layer2)
    p4 = p3.sequential(self.body.layer3)
    p5 = p4.sequential(self.body.layer4)
    return self.fpn([p3, p4, p5])

class ExtraFPNBlock:
  def __init__(self, in_channels, out_channels):
    self.p6 = Conv2dKaiming(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.p7 = Conv2dKaiming(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.use_P5 = in_channels == out_channels

  def __call__(self, p, c):
    p5, c5 = p[-1], c[-1]
    x = p5 if self.use_P5 else c5
    p6 = self.p6(x)
    p7 = self.p7(p6.relu())
    p.extend([p6, p7])
    return p

class FPN:
  def __init__(self, in_channels_list, out_channels, extra_blocks=None):
    self.inner_blocks, self.layer_blocks = [], []
    for in_channels in in_channels_list:
      self.inner_blocks.append(Conv2dKaiming(in_channels, out_channels, kernel_size=1))
      self.layer_blocks.append(Conv2dKaiming(out_channels, out_channels, kernel_size=3, padding=1))
    self.extra_blocks = ExtraFPNBlock(256, 256) if extra_blocks is None else extra_blocks

  def __call__(self, x):
    last_inner = self.inner_blocks[-1](x[-1])
    results = [self.layer_blocks[-1](last_inner)]
    for idx in range(len(x) - 2, -1, -1):
      inner_lateral = self.inner_blocks[idx](x[idx])

      # upsample to inner_lateral's shape
      (ih, iw), (oh, ow), prefix = last_inner.shape[-2:], inner_lateral.shape[-2:], last_inner.shape[:-2]
      eh, ew = math.ceil(oh / ih), math.ceil(ow / iw)
      inner_top_down = last_inner.reshape(*prefix, ih, 1, iw, 1).expand(*prefix, ih, eh, iw, ew).reshape(*prefix, ih*eh, iw*ew)[:, :, :oh, :ow]

      last_inner = inner_lateral + inner_top_down
      results.insert(0, self.layer_blocks[idx](last_inner))
    if self.extra_blocks is not None:
      results = self.extra_blocks(results, x)
    return results

if __name__ == "__main__":
  from extra.models.resnet import ResNeXt50_32X4D
  backbone = ResNeXt50_32X4D()
  retina = RetinaNet(backbone)
  retina.load_from_pretrained()