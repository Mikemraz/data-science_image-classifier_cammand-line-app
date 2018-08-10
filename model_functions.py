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
from preprocessing import find_key,predict_index_to_name

def do_deep_learning(model, trainloader, validationloader, epochs=7, lr=0.001, gpu_truth=True):
    epochs = epochs
    steps = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    print_every = 40
    device = ''
    if gpu_truth:
        device='cuda'
    else:
        device='cpu'
    model.to(device)      
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (images, labels) in enumerate(trainloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)            
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every== 0:      
                model.eval()
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validationloader, criterion)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))
                running_loss = 0
                model.train()

                
def build_model(arch,hidden_units):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(4096, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier    
    return model


def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:   
        images, labels = images.to('cuda'), labels.to('cuda')
        model = model.cuda()
        model.eval()
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


def save_model(checkpoint,save_dir='checkpoint.pth'):
    torch.save(checkpoint, save_dir)
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    other_information = {}
    other_information['epoch'] = checkpoint['epoch']
    other_information['class_to_idx'] = checkpoint['class_to_idx']
    get_arch = checkpoint['arch']
    get_hidden_units = checkpoint['hidden_units']
    model = build_model(get_arch, get_hidden_units)
    model.load_state_dict(checkpoint['state_dict'])
    return model,other_information    
    
    
def predict(image, model, topk=5,category_name='',gpu_truth=True, other_info={}):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = ''
    if gpu_truth:
        device= 'cuda'
    else:
        device = 'cpu'
    image = torch.unsqueeze(image, 0)
    model = model.to(device)
    image = image.to(device)
    image = image.float()
    output = model.forward(image)
    # need to normalize the probablity, so I take all outputs into account.
    prob, predictions = output.topk(102)
    predictions = predictions.cpu().numpy()
    predictions = predictions.reshape([-1,])
    prob = prob.cpu().detach().numpy()
    prob = prob.reshape([-1,])
    # get the exponential of logsoftmax, which is unnormalized softmax
    prob = np.exp(prob)
    sum_prob = np.sum(prob)
    # normalize
    prob = np.divide(prob,sum_prob)    
    prob_topk = prob[:topk]
    predictions_topk = predictions[:topk]
    if category_name=='':
        return predictions_topk, prob_topk
    else:
        mapped_predictions = predict_index_to_name(predictions_topk, category_name, other_info)    
        return mapped_predictions, prob_topk
    # TODO: Implement the code to predict the class from an image file