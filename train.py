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
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os

parser = argparse.ArgumentParser(description='This is a train file')

parser.add_argument('data_directory', action="store", nargs='?', default="flowers", help="dataset directory")

parser.add_argument('--save_dir', default="", help="checkpoint save", dest="save_directory")
parser.add_argument('--arch',  default="vgg16", choices=['vgg13', 'vgg19'],help="you can only choose VGG family such as vgg13 or vgg19", dest="architecture")


parser.add_argument('--learning_rate', default="0.001", type=float, help="Set Learning rate",  dest="learning_rate")
parser.add_argument('--hidden_units', nargs=3, default=[1024, 512, 256], type=int, dest="hidden_units")
parser.add_argument('--epochs',default=1, type=int, help="choose epochs", dest="epochs")
parser.add_argument('--gpu', default=False, help="GPU", dest="gpu")


args = parser.parse_args()
dataDir =  args.data_directory
saveDir =  args.save_directory
architecture =  args.architecture
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
if args.gpu and torch.cuda.is_available(): 
	arg_gpu = args.gpu
else:
    arg_gpu = False
    print('We will use cpu becuase of missing GPU')

print(f"dir dataset: {dataDir}, save dir: {saveDir}, Arc: {architecture}\n learning rate: {lr}, hidden units: {hidden_units}, number of epochs: {epochs}, GPU: arg_gpu")
    
data_dir = dataDir
train_dir = dataDir + '/train'
valid_dir = dataDir + '/valid'
test_dir = dataDir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

if architecture == 'vgg13':
	print('wait'+str(architecture))
	model = models.vgg13(pretrained=True)
	print(f'Model = {architecture}')
	print(model)
else:
	print('Wait'+str(architecture))
	model = models.vgg19(pretrained=True)
	print(f'Model = {architecture}')
	print(model)

device = torch.device("cuda" if arg_gpu else "cpu")

print(f'Using {device}')

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(25088, hidden_units[0]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units[0], hidden_units[1]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units[1], hidden_units[2]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),                                 
                                 nn.Linear(hidden_units[2], 102),
                                 nn.LogSoftmax(dim=1))


print(model.classifier)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

model.to(device);
print('wait model is training')

epochs = epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)                    
                    valid_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
print('Validatation')

total = 0
length_all = 0
accuracy_all = 0
batch = 0
for inputs, labels in testloader:
    batch += 1    
    inputs, labels = inputs.to(device), labels.to(device) 
    accuracy = 0
    model.eval()
    with torch.no_grad():
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        total += torch.sum(equals)
        length_all += len(equals)
        accuracy_all = total.item()/total_length
    print(f"Batch {batch}.. "
          f"Accuracy: {accuracy*100:.4f}%.. "
          f"Total Accuracy: {total_accuracy*100:.4f}%")

    model.train()
    
    
print('Saving chekpoint..')

if save_dir:
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_dir = save_dir + '/checkpoint.pth'
else:
	save_dir = 'checkpoint.pth'


model.class_to_idx = train_data.class_to_idx
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epoch': epochs,
              'classifier': model.classifier,
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'learning_rate': arg_lr,
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, save_dir)

print('checkpoint saved success\n')

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    epoch = checkpoint['epoch']
    
    return model, optimizer, input_size, output_size, epoch 


model, opt, input_size, output_size, epoch  = load_checkpoint(save_dir)


print('Saved model:')
print(model)
