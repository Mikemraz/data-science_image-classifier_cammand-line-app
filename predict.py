import torch
from torchvision.utils import save_image
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
from preprocessing import process_image,find_key,predict_index_to_name
from model_functions import build_model,load_checkpoint,predict
import argparse

parser = argparse.ArgumentParser(description='use trained model to predict class of a image ')
parser.add_argument('path_to_image', help='the path of image')
parser.add_argument('model_directory', help='the directory of model that has been trained')
parser.add_argument('--top_K', type=int, default=3, help='return the top K most possible classes')
parser.add_argument('--category_names', default='' ,help='output the actual class name')
parser.add_argument('--gpu', action='store_true',default=False, help='the training mode')
args = parser.parse_args()

model, other_info = load_checkpoint(args.model_directory)
image = process_image(args.path_to_image)
classes, probs = predict(image, model, topk=args.top_K, category_name=args.category_names, gpu_truth=args.gpu, other_info=other_info)
print(classes)
print(probs)