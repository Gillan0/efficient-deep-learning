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

import torchprofile

def score(p_s, p_u, q_w, q_a, w, f):
    p = ((1 - p_s - p_u) * w * q_w / 32) / (5.6 * 10 ** 6)
    ops = ((1 - p_s) * max(q_w, q_a) * f / 32) / (2.8 * 10 ** 8)

    print(f"p : {p:.4f}\nops : {ops:.4f}")

    return p + ops

def count_params(model, layer_type):
    total = 0
    for m in model.modules():
        if isinstance(m, layer_type):
            total += sum(p.numel() for p in m.parameters())
    return total

def test(net):
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)#.half()
            targets =  targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    avg_loss = test_loss / len(testloader)
    acc = 100. * correct / total
    return loss, acc
  
if __name__ == "__main__":
    linear_percentage = 0.60
    convolutional_percentage = 0.83

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = ResNet18()
    net = net.to(device) 
    #net.half()
    BITS = 32#16


    MODEL_DIR ="./src/lab3/og_models/"
    MODEL_NAME = "cosine-mixup-60-83-pruned"


    checkpoint = torch.load(MODEL_DIR + MODEL_NAME)
    net.load_state_dict(checkpoint['net'])
    criterion = nn.CrossEntropyLoss()    



    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    conv_params = count_params(net, torch.nn.Conv2d)
    linear_params = count_params(net, torch.nn.Linear)

    w = conv_params + linear_params
    
    input_tensor = torch.randn(1, 3, 32, 32)#.half()  
    f = torchprofile.profile_macs(net, input_tensor)

    print(f"Conv params: {conv_params}")
    print(f"Conv pruning: {convolutional_percentage}")
    
    print(f"Linear params: {linear_params}")
    print(f"Linear pruning: {linear_percentage}")
    
    print(f"Total params in conv+linear: {w}")
    p_u = (conv_params * convolutional_percentage + linear_params * linear_percentage) / (conv_params + linear_params)
    print(f"Total pruning percentage: { p_u:.4f}")

    print(f"MACS : {f}")
    score = score(p_s=0, p_u=p_u, q_w=BITS, q_a=BITS, w=w, f=f)
    print(f"Score : {score:.4f}")
    loss, acc = test(net)