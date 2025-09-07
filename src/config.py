"""
Configuration file for SqueezeNet CUDA implementation
"""
import os

# Random seed for reproducibility
SEED = 42

# Data paths
DIR_ROOT = '/kaggle/input/tomato-diseases'
DIR_TRAIN = os.path.join(DIR_ROOT, 'train')
DIR_TEST = os.path.join(DIR_ROOT, 'test')

# Training parameters
BATCH_SIZE = 32
SHUFFLE = True
LEARNING_RATE = 0.0008
EPOCHS = 5

# CUDA parameters
TPB_CONV = 16
TPB_POOL = 32
TPB_SOFTMAX = 10
TPB_BACKWARD = 256

# Model parameters
NUM_CLASSES = 10
INPUT_SIZE = 224

# File paths
MODEL_SAVE_PATH = '/kaggle/working/model_data_SGD_epoch_30.npz'
MODEL_LOAD_PATH = "/kaggle/input/epoch_30_sgd/other/default/1/model_data_SGD_epoch_25.npz"
