import os
import torch
import numpy as np
import pandas as pd
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
import matplotlib.pyplot as plt

EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'}
DATA_PATH = 'C:/Users/Saransh/Desktop/Research Team/data/Combined/Emotions/'
SAMPLE_RATE = 48000

