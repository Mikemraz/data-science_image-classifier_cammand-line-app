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

def get_dataloaders(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return trainloader, validationloader, testloader, train_data


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_loader = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor()])
    
    im = Image.open(image)
    im = img_loader(im).float()
    np_image = np.array(im)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1)) 
    py_image = torch.from_numpy(np_image)
    return py_image

def find_key(value, other_info):
    for k,v in other_info['class_to_idx'].items():
        if v==value:
            return k
        
        
def predict_index_to_name(predictions, cat_to_name, other_info):
    dict_cat_to_name = {}
    with open(cat_to_name, 'r') as f:
        dict_cat_to_name = json.load(f)
    new_index = map(lambda p: find_key(p, other_info), predictions)
    names = [dict_cat_to_name[str(index)] for index in new_index]
    return names
