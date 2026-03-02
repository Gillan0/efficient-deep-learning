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

import torchprofile

def score(p_s, p_u, q_w, q_a, w, f):
    p = ((1 - p_s - p_u) * w * q_w / 32) / (5.6 * 10 ** 6)
    ops = ((1 - p_s) * max(q_w, q_a) * f / 32) / (2.8 * 10 ** 8)

    return p + ops

def count_params(model, layer_type):
    total = 0
    for m in model.modules():
        if isinstance(m, layer_type):
            total += sum(p.numel() for p in m.parameters())
    return total

def count_macs(model, input_size=(1, 3, 32, 32)):
    macs = 0

    def conv_hook(self, input, output):
        # input[0] shape: [batch, in_channels, H, W]
        batch_size, in_c, H, W = input[0].shape
        out_c, out_h, out_w = output.shape[1:4]
        kernel_h, kernel_w = self.kernel_size
        groups = self.groups
        # MACs per conv layer
        layer_macs = batch_size * out_c * out_h * out_w * (in_c // groups) * kernel_h * kernel_w
        nonlocal macs
        macs += layer_macs

    def linear_hook(self, input, output):
        batch_size = input[0].shape[0]
        in_features = self.in_features
        out_features = self.out_features
        layer_macs = batch_size * in_features * out_features
        nonlocal macs
        macs += layer_macs

    hooks = []

    # register hooks
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    # run a dummy forward
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)
    model.eval()
    with torch.no_grad():
        model(x)

    # remove hooks
    for h in hooks:
        h.remove()

    return macs

    
if __name__ == "__main__":
    linear_percentage = 0.6
    convolutional_percentage = 0.83 

    net = ResNet18()

    conv_params = count_params(net, torch.nn.Conv2d)
    linear_params = count_params(net, torch.nn.Linear)

    w = conv_params + linear_params
    f = count_macs(net)

    

    print(f"Conv params: {conv_params}")
    print(f"Conv pruning: {convolutional_percentage}")
    
    print(f"Linear params: {linear_params}")
    print(f"Linear pruning: {linear_percentage}")
    
    print(f"Total params in conv+linear: {w}")
    p_u = (conv_params * convolutional_percentage + linear_params * linear_percentage) / (conv_params + linear_params)
    print(f"Total pruning percentage: { p_u:.4f}")

    print(f"MACS : {f}")
    score = score(p_s=0, p_u=0, q_w=16, q_a=16, w=w, f=f)
    print(f"Score : {score}")