from collections import OrderedDict
import unicodedata
import numpy as np
from tinygrad.nn import state
from tinygrad.tensor import Tensor, dtypes
from tinygrad.helpers import getenv

#
# checkpointing utils
#

def invert_dict(d): return {v: k for k, v in reversed(d.items())}
def dedup_dict(d): return invert_dict(invert_dict(d))
# store each tensor into the first key it appears in
def get_training_state(model, optimizer, scheduler):
  # hack: let get_state_dict walk the tree starting with model, so that the checkpoint keys are
  # readable and can be loaded as a model for eval
  train_state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
  return dedup_dict(state.get_state_dict(train_state))
def load_training_state(model, optimizer, scheduler, state_dict):
  # use fresh model to restore duplicate keys
  train_state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
  big_dict = state.get_state_dict(train_state)
  # hack: put back the dupes
  dupe_names = {}
  for k, v in big_dict.items():
    if v not in dupe_names:
      dupe_names[v] = k
      assert k in state_dict
    state_dict[k] = state_dict[dupe_names[v]]
  # scheduler contains optimizer and all params, load each weight only once
  scheduler_state = {'scheduler': scheduler}
  state.load_state_dict(scheduler_state, state_dict)

def gaussian_kernel(n, std):
  from scipy import signal
  gaussian_1d = signal.windows.gaussian(n, std)
  gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
  gaussian_3d = np.outer(gaussian_2d, gaussian_1d)
  gaussian_3d = gaussian_3d.reshape(n, n, n)
  gaussian_3d = np.cbrt(gaussian_3d)
  gaussian_3d /= gaussian_3d.max()
  return gaussian_3d

def prepare_arrays(image, roi_shape=(128, 128, 128)):
  assert len(roi_shape) == 3 and any(roi_shape)
  image_shape = list(image.shape[2:])
  result = np.zeros((1, 3, *image_shape), dtype=image.dtype)
  norm_map = np.zeros_like(result)
  norm_patch = gaussian_kernel(roi_shape[0], 0.125 * roi_shape[0]).astype(norm_map.dtype)
  return result, norm_map, norm_patch

def get_slice(image, roi_shape=(128, 128, 128), overlap_factor=0.5):
  assert len(roi_shape) == 3 and any(roi_shape)
  assert 0 < overlap_factor < 1
  image_shape, dim = list(image.shape[2:]), len(image.shape[2:])
  strides = [int(roi_shape[i] * (1 - overlap_factor)) for i in range(dim)]
  size = [(image_shape[i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
  for i in range(0, strides[0] * size[0], strides[0]):
    for j in range(0, strides[1] * size[1], strides[1]):
      for k in range(0, strides[2] * size[2], strides[2]):
        yield i, j, k

def _get_best_indices(logits, n_best_size):
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
  return list(map(lambda x: x[0], index_and_score))[:n_best_size]

def _is_punctuation(char):
  if (cp := ord(char)) in range(33, 48) or cp in range(58, 65) or cp in range(91, 97) or cp in range(123, 127):
    return True
  return unicodedata.category(char).startswith("P")

def _is_whitespace(char):
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  return unicodedata.category(char) == "Zs"

def _is_control(char):
  if char == "\t" or char == "\n" or char == "\r":
    return False
  return unicodedata.category(char).startswith("C")

def _run_split_on_punc(text):
  if text in ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"):
    return [text]
  start_new_word = True
  output = []
  for i in range(len(text)):
    if _is_punctuation(char := text[i]):
      output.append([char])
      start_new_word = True
    else:
      if start_new_word:
        output.append([])
      start_new_word = False
      output[-1].append(char)
  return ["".join(x) for x in output]

def _run_strip_accents(text):
  output = []
  for char in unicodedata.normalize("NFD", text):
    if unicodedata.category(char) != "Mn":
      output.append(char)
  return "".join(output)

def _clean_text(text):
  output = []
  for char in text:
    if not ((cp := ord(char)) == 0 or cp == 0xfffd or _is_control(char)):
      output.append(" " if _is_whitespace(char) else char)
  return "".join(output)

def _get_final_text(pred_text, orig_text):
  def _strip_spaces(text):
    ns_text = ""
    ns_to_s_map = OrderedDict()
    for i, c in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_text)] = i
      ns_text += c
    return ns_text, ns_to_s_map

  orig_tokens = _clean_text(orig_text).strip().split()
  split_tokens = []
  for token in orig_tokens:
    if token not in ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"):
      token = token.lower()
      token = _run_strip_accents(token)
    split_tokens.extend(_run_split_on_punc(token))

  tok_text = " ".join(" ".join(split_tokens).strip().split())
  start_position = tok_text.find(pred_text)
  if start_position == -1:
    return orig_text
  end_position = start_position + len(pred_text) - 1

  orig_ns_text, orig_ns_to_s_map = _strip_spaces(orig_text)
  tok_ns_text, tok_ns_to_s_map = _strip_spaces(tok_text)
  if len(orig_ns_text) != len(tok_ns_text):
    return orig_text
  tok_s_to_ns_map = {v: k for k, v in tok_ns_to_s_map.items()}

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    if (ns_start_position := tok_s_to_ns_map[start_position]) in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]
  if orig_start_position is None:
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    if (ns_end_position := tok_s_to_ns_map[end_position]) in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]
  if orig_end_position is None:
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text

def get_bert_qa_prediction(features, example, start_end_logits):
  prelim_predictions = []
  for i, feature in enumerate(features):
    for start_index in _get_best_indices(start_end_logits[i][0], 20):
      for end_index in _get_best_indices(start_end_logits[i][1], 20):
        if start_index >= len(feature["tokens"]) or end_index >= len(feature["tokens"]):
          continue
        if start_index not in feature["token_to_orig_map"] or end_index not in feature["token_to_orig_map"]:
          continue
        if not feature["token_is_max_context"].get(start_index, False):
          continue
        if end_index < start_index or end_index - start_index + 1 > 30:
          continue

        prelim_predictions.append({
          "feature_index": i,
          "start_index": start_index,
          "end_index": end_index,
          "start_logit": start_end_logits[i][0, start_index],
          "end_logit": start_end_logits[i][1, end_index]
        })
  predictions = sorted(prelim_predictions, key=lambda x: (x["start_logit"] + x["end_logit"]), reverse=True)

  if len(predictions) > 0:
    feature = features[predictions[0]["feature_index"]]
    tok_tokens = feature["tokens"][predictions[0]["start_index"]:(predictions[0]["end_index"] + 1)]
    orig_doc_start = feature["token_to_orig_map"][predictions[0]["start_index"]]
    orig_doc_end = feature["token_to_orig_map"][predictions[0]["end_index"]]
    orig_tokens = example["context"][orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens).replace(" ##", "").replace("##", "")
    tok_text = " ".join(tok_text.strip().split())
    orig_text = " ".join(orig_tokens)
    return _get_final_text(tok_text, orig_text)
  return "empty"

def get_mlperf_bert_config():
  """Config is BERT-large"""
  return {
    "attention_probs_dropout_prob": 0.1,
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "max_position_embeddings": 512,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "type_vocab_size": 2,
    "vocab_size": 30522
  }

def get_mlperf_bert_model(checkpoint_path:str):
  from extra.models import bert
  from examples.mlperf.initializers import LinearBert, EmbeddingBert, LayerNormBert

  bert.Linear = LinearBert
  bert.Embedding = EmbeddingBert 
  bert.LayerNorm = LayerNormBert

  from extra.models.bert import BertForPretraining
  config = get_mlperf_bert_config()
  if getenv("DISABLE_DROPOUT", 0):
    config["hidden_dropout_prob"] = config["attention_probs_dropout_prob"] = 0.0
  return BertForPretraining(**config).load_from_pretrained(checkpoint_path)

def get_data_bert(GPUS:list[str], it):
  data: dict[str, Tensor] = next(it)
  for key in data.keys(): data[key].shard_(GPUS, axis=0)
  return data

def get_fake_data_bert(GPUS:list[str], BS:int):
  return {
    "input_ids": Tensor.zeros((BS, 512), dtype=dtypes.float32).contiguous().shard_(GPUS, axis=0),
    "input_mask": Tensor.zeros((BS, 512), dtype=dtypes.default_float).contiguous().shard_(GPUS, axis=0),
    "segment_ids": Tensor.zeros((BS, 512), dtype=dtypes.float32).contiguous().shard_(GPUS, axis=0),
    "masked_lm_positions": Tensor.zeros((BS, 512), dtype=dtypes.float32).contiguous().shard_(GPUS, axis=0),
    "masked_lm_ids": Tensor.zeros((BS, 512), dtype=dtypes.float32).contiguous().shard_(GPUS, axis=0),
    "masked_lm_weights": Tensor.zeros((BS, 512), dtype=dtypes.float32).contiguous().shard_(GPUS, axis=0),
    "next_sentence_labels": Tensor.zeros((BS, 1), dtype=dtypes.float32).contiguous().shard_(GPUS, axis=0),
  }
  
import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
import os
from collections import defaultdict
import sys
import mmap
import pickle

class COCO:
    def __init__(self, annotation_file=None):
        self.dataset = {}
        self.anns = {}
        self.cats = {}
        self.imgs = {}
        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)
        
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            
            # Check if a pickled index file exists
            index_file = annotation_file + '.pickle'
            if os.path.exists(index_file):
                with open(index_file, 'rb') as f:
                    index_data = pickle.load(f)
                self.anns = index_data['anns']
                self.cats = index_data['cats']
                self.imgs = index_data['imgs']
                self.imgToAnns = index_data['imgToAnns']
                self.catToImgs = index_data['catToImgs']
                print(f'Loaded indexed data from {index_file}')
            else:
                # If no index file exists, create it
                with open(annotation_file, 'r+b') as f:
                    mm = mmap.mmap(f.fileno(), 0)
                    self.dataset = json.loads(mm.read().decode('utf-8'))
                assert isinstance(self.dataset, dict), 'annotation file format {} not supported'.format(type(self.dataset))
                self.createIndex()
                
                # Save the index to a pickle file for future use
                index_data = {
                    'anns': self.anns,
                    'cats': self.cats,
                    'imgs': self.imgs,
                    'imgToAnns': self.imgToAnns,
                    'catToImgs': self.catToImgs
                }
                with open(index_file, 'wb') as f:
                    pickle.dump(index_data, f)
                print(f'Saved indexed data to {index_file}')
            
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def createIndex(self):
        print('creating index...')
        
        # Create anns, imgs, and cats indices
        for ann in self.dataset['annotations']:
            self.imgToAnns[ann['image_id']].append(ann)
            self.anns[ann['id']] = ann
            if 'category_id' in ann:
                self.catToImgs[ann['category_id']].append(ann['image_id'])

        for img in self.dataset['images']:
            self.imgs[img['id']] = img

        for cat in self.dataset['categories']:
            self.cats[cat['id']] = cat

        print('index created!')

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        imgIds = set(imgIds if isinstance(imgIds, list) else [imgIds])
        catIds = set(catIds if isinstance(catIds, list) else [catIds])

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.anns.values()
        else:
            if not imgIds:
                anns = self.anns.values()
            else:
                anns = [ann for imgId in imgIds for ann in self.imgToAnns[imgId]]
            anns = [ann for ann in anns if not catIds or ann['category_id'] in catIds]
            if areaRng:
                anns = [ann for ann in anns if areaRng[0] < ann['area'] < areaRng[1]]
        if iscrowd is not None:
            anns = [ann for ann in anns if ann['iscrowd'] == iscrowd]
        return [ann['id'] for ann in anns]

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        catNms = set(catNms if isinstance(catNms, list) else [catNms])
        supNms = set(supNms if isinstance(supNms, list) else [supNms])
        catIds = set(catIds if isinstance(catIds, list) else [catIds])

        if not catNms and not supNms and not catIds:
            return list(self.cats.keys())
        cats = self.cats.values()
        if catNms:
            cats = [cat for cat in cats if cat['name'] in catNms]
        if supNms:
            cats = [cat for cat in cats if cat['supercategory'] in supNms]
        if catIds:
            cats = [cat for cat in cats if cat['id'] in catIds]
        return [cat['id'] for cat in cats]

    def getImgIds(self, imgIds=[], catIds=[]):
        imgIds = set(imgIds if isinstance(imgIds, list) else [imgIds])
        catIds = set(catIds if isinstance(catIds, list) else [catIds])

        if not imgIds and not catIds:
            return list(self.imgs.keys())
        if not catIds:
            return list(imgIds & set(self.imgs.keys()))
        imgIds = set.union(*[set(self.catToImgs[catId]) for catId in catIds]) if not imgIds else imgIds
        return list(imgIds & set(self.imgs.keys()))

    def loadAnns(self, ids=[]):
        return [self.anns[id] for id in ids] if isinstance(ids, list) else [self.anns[ids]]


    def loadCats(self, ids=[]):
        return [self.cats[id] for id in ids] if isinstance(ids, list) else [self.cats[ids]]

    def loadImgs(self, ids=[]):
        return [self.imgs[id] for id in ids] if isinstance(ids, list) else [self.imgs[ids]]


    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'

        if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann

