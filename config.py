import os
from yacs.config import CfgNode as CN

DATA_DIR = '/data/dl'

_C = CN()
_C.EXP_NM = 'test'
_C.EXP_DIR = 'dl_trainer'

# training params
_C.TRAIN = CN()
_C.TRAIN.NUM_EPOCHS = 5
_C.TRAIN.GRAD_ACCUM_STEPS = 1
_C.TRAIN.EVAL_STEPS = 1000
_C.TRAIN.CP_DIR = 'checkpoints'
_C.TRAIN.DEVICE = 'gpu'
_C.TRAIN.SAVE_CP = True
_C.TRAIN.EVAL_ONLY = False

# model params
_C.TRAIN.MODEL = CN()
_C.TRAIN.MODEL.MODEL_PATH = 'bert-base-uncased'

# optimizer params
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.LR = 3e-5

# learning rate scheduler params
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'constant'
# scheduler types: ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
_C.TRAIN.LR_SCHEDULER.WARMUP_RATIO = 0.10

# dataloader params
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 8
_C.DATALOADER.VAL_BATCH_SIZE = 8
_C.DATALOADER.TEST_BATCH_SIZE = 8

# dataset params
_C.DATASETS = CN()

# train data
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.CSV_PATH = os.path.join(DATA_DIR, 'train_sample.csv')

# val data
_C.DATASETS.VAL = CN()
_C.DATASETS.VAL.CSV_PATH = os.path.join(DATA_DIR, 'valid_sample.csv')

# test data
_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.CSV_PATH = os.path.join(DATA_DIR, 'valid_sample.csv')

# tensorboard writer
_C.WRITER = CN()

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
