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
from preprocessing import get_dataloaders
from model_functions import build_model,validation,do_deep_learning,save_model
import argparse

parser = argparse.ArgumentParser(description='train a image classifier')
parser.add_argument('data_directory', help='the directory of data that the model is trained on')
parser.add_argument('--save_dir', default='checkpoint.pth', help='the directory to save the model')
parser.add_argument('--arch', default='vgg13',help='the model architecture used for training')
parser.add_argument('--learning_rate', type=float, default=0.001 ,help='the learning rate for training')
parser.add_argument('--hidden_units', type=int, default=512 ,help='the hidden units in fully connected layer')
parser.add_argument('--epochs', type=int, default=7 ,help='the number of epochs for training')
parser.add_argument('--gpu', action='store_true',default=False, help='the training mode')
args = parser.parse_args()

trainloader, validationloader, testloader, train_data = get_dataloaders(args.data_directory)

model = build_model(args.arch,args.hidden_units)
do_deep_learning(model, trainloader, validationloader, epochs=args.epochs, lr=args.learning_rate, gpu_truth=args.gpu)
checkpoint = {'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'epoch':args.epochs,
              'arch':args.arch,
              'hidden_units':args.hidden_units}
save_model(checkpoint,save_dir=args.save_dir)