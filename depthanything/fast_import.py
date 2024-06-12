from depth_anything.dpt import DepthAnything
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
encoder = 'vitb' # or 'vitb', 'vits'
depth_anything_model = DepthAnything(model_configs[encoder]).cuda().eval()