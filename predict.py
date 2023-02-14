
#Author: Manzoor Ali

# Issues: I am unable to use GPU

# sources used:
    # https://github.com/ErkanHatipoglu
    # https://github.com/CaterinaBi
    # https://discuss.pytorch.org/t/attributeerror-numpy-ndarray-object-has-no-attribute-numpy/42062/3
    # https://discuss.pytorch.org/t/how-to-represent-class-to-idx-map-for-custom-dataset-in-pytorch/37510
    # https://discuss.pytorch.org/t/cuda-runtime-error-2-out-of-memory-at-opt-conda-conda-bld-pytorch-1518238409320-work-torch-lib-thc-generic-thcstorage-cu-58/17823
    # https://towardsdatascience.com/load-that-checkpoint-51142d44fb5d
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    # Also help from the classroom assignments


import argparse
import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import random
from PIL import Image
import numpy as np


def load_Model(filepath):
    model = models.vgg16(pretrained=True)
    checkpoint = torch.load(filepath)
    lr = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    epoch = checkpoint['epoch']
    
    return model, optimizer, input_size, output_size, epoch


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    my_image = Image.open(image)
    my_image = my_image.resize((256, 256))

    (left, upper, right, lower) = (16, 16, 240, 240)
    my_image = my_image.crop((left, upper, right, lower))
    np_image = np.array(my_image)/255
    np_image = (np_image - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    return np_image.transpose(2, 0, 1)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device);
    model.eval()
    with torch.no_grad():
        my_image = process_image(image_path)
        my_image = torch.from_numpy(my_image).unsqueeze(0)
        my_image = my_image.to(device);
        my_image = my_image.float()
        model = model.to(device);
        logps = model.forward(my_image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        return top_p, top_class 

parser = argparse.ArgumentParser(description="Predict the iamges using trained model") 
parser.add_argument('path_to_image', nargs='?', default='./flowers/test/19/image_06186.jpg')
parser.add_argument('path_to_checkpoint',  nargs='?', default='checkpoint.pth')
parser.add_argument('--top_k', action="store", default=1, type=int, help="enter number of guesses", dest="top_k")
parser.add_argument('--category_names', action="store", default="cat_to_name.json", dest="category_names")
parser.add_argument('--gpu', action="store_true", default=False, dest="gpu")

args = parser.parse_args()

arg_path_to_image =  args.path_to_image
arg_path_to_checkpoint = args.path_to_checkpoint
arg_top_k =  args.top_k
arg_category_names =  args.category_names
if args.gpu and torch.cuda.is_available(): 
	arg_gpu = args.gpu
elif args.gpu:
	arg_gpu = False
	print('GPU is not available, will use CPU...')
else:
	arg_gpu = args.gpu
device = torch.device("cuda" if arg_gpu else "cpu")
print('We will use cpu becuase of missing GPU')

with open(arg_category_names, 'r') as f:
    cat_to_name = json.load(f)

model, optimizer, input_size, output_size, epoch  = load_Model(arg_path_to_checkpoint)
model.eval()

idx_to_class = {v:k for k, v in model.class_to_idx.items()}
print(arg_path_to_image)
probs, classes = predict('{}'.format(arg_path_to_image), model, arg_top_k)

for count in range(arg_top_k):
     print(f'{cat_to_name[idx_to_class[classes[0, count].item()]]} ...........{probs[0, count].item()}')
