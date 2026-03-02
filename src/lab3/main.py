'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import *
from utils import progress_bar

from dataloader import testloader, trainloader
#from debug_dataloader import trainloader

# Binary Connect : 
import binaryconnect
from mixup import mixup_data, mixup_criterion


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--name', default="ckpt.pth", type=str, help='Model ID')
parser.add_argument('--log', default='training.log', type=str, help='Log file name (.log)')
parser.add_argument('--epoch', default=50, type=int, help='Number of epochs')
parser.add_argument('--model', default="cosine", type=str, help="Model of choice : 'cosine', 'plateau', 'adam'")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
NB_EPOCH = args.epoch

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = binaryconnect.BC(ResNet18()) 
net.model = net.model.to(device) 

if not(args.name):
    raise Exception("No model name")

print('==> Resuming from checkpoint..')
assert os.path.isdir('./src/lab3/og_models/'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./src/lab3/og_models/' + args.name)
net.model.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = 0
criterion = nn.CrossEntropyLoss()

if torch.cuda.device_count() > 1:
    net.model = torch.nn.DataParallel(net.model)

if args.model == "adam":
    optimizer = optim.Adam(net.model.parameters(), 
                       lr=args.lr, 
                       weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

elif args.model == "plateau":
    optimizer = optim.SGD(net.model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

else:
    optimizer = optim.SGD(net.model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-6)        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NB_EPOCH)

# Log file 
# Initialize log file
with open("./src/lab3/logs/" + args.log, "w") as f:
    f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    
    net.model.train()
    net.binarization()

    alpha = 1.0 # Mixup param

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, device)


        optimizer.zero_grad()

        net.binarization() 
        outputs = net.forward(inputs)
        net.restore() 


        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        net.clip()
        net.save_params()

        progress_bar(batch_idx, len(trainloader))

def test(epoch):
    global best_acc
    net.model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net.forward(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    avg_loss = test_loss / len(testloader)
    acc = 100. * correct / total

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './src/lab3/checkpoint/' + args.name)
        best_acc = acc

    return avg_loss, acc

best_acc = 0.0
for epoch in range(start_epoch, start_epoch + NB_EPOCH):
    train(epoch)
    test_loss, test_acc = test(epoch)
    if args.model == "cosine":
        scheduler.step()
    else:
        scheduler.step(train_loss)

    with open("./src/lab3/logs/" + args.log, "a") as f:
        f.write(f"{epoch},NaN,NaN,{test_loss:.4f},{test_acc:.2f}\n")