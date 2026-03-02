import time

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

from dataloader import testloader

import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--load', default="cosine_clone", type=str, help="Model to load")
parser.add_argument('--half', action='store_true', help="Use half precision model (FP16)")
parser.add_argument('--quarter', action='store_true', help="Use quarter precision model (FP8)")

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


if args.load:
    # Load Model
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    net = ResNet18()
    net = net.to(device)

    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./src/lab3/og_models/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./src/lab3/og_models/' + args.load)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = 0
    criterion = nn.CrossEntropyLoss()


    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

else:
    raise Exception("No loaded model")

if args.half:
    net.half()


# Log file 
# Initialize log file
with open("./src/lab3/logs/half_" + args.load, "w") as f:
    f.write("test_loss,test_acc\n")

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_inference_time = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Half tensor inputs when half data
            if args.half:
                inputs = inputs.half()

            # Timer start
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = net(inputs)
            
            # Timer end
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            total_inference_time += (end - start)

            # Loss computation
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    avg_loss = test_loss / len(testloader)
    acc = 100. * correct / total

    return avg_loss, acc, total_inference_time


def model_size(net):
    param_size = 0
    for param in net.parameters():
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in net.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2

test_loss, test_acc, time = test()

print(f"Test Loss : {test_loss};\nTest Acc : {test_acc}")

print(f"Argument size check : ", next(net.parameters()).dtype)
print(f"Model size (MB) : {model_size(net)}")
print(f"Model time - Total inference time : {time}")


with open("./src/lab3/logs/half_" + args.load, "a") as f:
    f.write(f"{test_loss},{test_acc}\n")