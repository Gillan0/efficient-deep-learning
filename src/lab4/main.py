'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.nn.utils.prune as prune


import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import *
from utils import progress_bar

from dataloader import testloader, trainloader
#from debug_dataloader import trainloader

from mixup import mixup_data, mixup_criterion


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--name', default="ckpt.pth", type=str, help='Model ID')
parser.add_argument('--retrain', action="store_true", help="Retrain the model after pruning")
parser.add_argument('--epoch', default=10, type=int, help='Number of epochs')
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
net = ResNet18()
net = net.to(device) 

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if not(args.name):
    raise Exception("No model name")

assert os.path.isdir('./src/lab1/checkpoint/'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(f'./src/lab1/checkpoint/{args.name}', weights_only=True)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

NB_EPOCH = args.epoch

criterion = nn.CrossEntropyLoss()
def test(epoch, linear_perc, conv_perc):
    global best_acc
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')

        if isinstance(net, torch.nn.DataParallel):
            model_to_save = net.module
        else:
            model_to_save = net

        state = {
            'net': model_to_save.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if args.retrain:
            torch.save(state, f"./src/lab4/checkpoint/{args.name}_pruned_retrained_{linear_perc}_{conv_perc}")
        else:
            torch.save(state, f"./src/lab4/checkpoint/{args.name}_pruned_{linear_perc}_{conv_perc}")
        best_acc = acc

    return avg_loss, acc


def train(epoch):
    print('\nEpoch: %d' % epoch)
    
    net.train()

    # train_loss = 0
    # correct = 0
    # total = 0

    alpha = 1.0 # Mixup param

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()


        progress_bar(batch_idx, len(trainloader))


def pruning(net, amount_linear=0.0, amount_conv=0.0):

    for idx, module in enumerate(net.modules()): 
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount_conv)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount_linear)

    return test(0, amount_linear, amount_conv)

if __name__ == "__main__":

    
    for loop_linear in range(60, 70, 10):
        for loop_conv in range(95, 96, 1):

            amount_linear = loop_linear /100
            amount_conv = loop_conv / 100

            net = ResNet18()
            net = net.to(device) 

            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True
            
            net.load_state_dict(checkpoint['net'])

            best_acc = 0.0

            print(f"Linear pruning : {amount_linear }; Convolutional pruning : {amount_conv }")
            pruned_loss, pruned_acc = pruning(net, amount_linear , amount_conv )


            if args.retrain:  
                with open(f"./src/lab4/logs/{args.name}_pruned_retrained_{loop_linear}_{loop_conv}.log", "w") as f:
                    f.write("epoch,linear_perc,conv_perc,pruned_loss,pruned_acc,test_loss,test_acc\n")

            else:
                with open(f"./src/lab4/logs/{args.name}_pruned_{loop_linear}_{loop_conv}.log", "w") as f:
                    f.write("linear_perc,conv_perc,pruned_loss,pruned_acc,\n")

            if args.retrain:
                if args.model == "adam":
                    optimizer = optim.Adam(net.parameters(), 
                                    lr=args.lr, 
                                    weight_decay=5e-6)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NB_EPOCH)

                elif args.model == "plateau":
                    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=5e-6)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

                else:
                    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=5e-6)        
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NB_EPOCH)

                for epoch in range(args.epoch):
                    train(epoch)
                    test_loss, test_acc = test(epoch, loop_linear, loop_conv)

                    if args.model == "cosine" or args.model == "adam" :
                        scheduler.step()
                    else:
                        scheduler.step(train_loss)

                    with open(f"./src/lab4/logs/{args.name}_pruned_retrained_{loop_linear}_{loop_conv}.log", "a") as f:
                        f.write(f"{epoch},{amount_linear },{amount_conv },{pruned_loss:.4f},{pruned_acc:.2f},{test_loss:.4f},{test_acc:.2f}\n")
            else:
                with open(f"./src/lab4/logs/{args.name}_pruned_{loop_linear}_{loop_conv}.log", "a") as f:
                    f.write(f"{amount_linear },{amount_conv },{pruned_loss:.4f},{pruned_acc:.2f}\n")
