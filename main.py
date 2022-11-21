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
import pickle

from resnet import *
from utils import progress_bar, save_plot_over_training


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=50, type=int, help='num epochs')
parser.add_argument('--mname', required=True, type=str, help='unique model name')
parser.add_argument('--optimz', default='sgd', type=str, help='optimizer: sgd, adam, adadelta, rmsprop', choices=['sgd', 'adam', 'adadelta', 'rmsprop'])
parser.add_argument('--model', default='ResNet14', type=str, help='model: ResNet10, ResNet14, ResNet14_v2', choices=['ResNet10', 'ResNet14', 'ResNet14_v2'])
parser.add_argument('--batch_size', default=128, type=int, help="The batch size for training data")
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--do_annealing', action='store_true', help="Whether to use cosine annealing or not")
parser.add_argument('--overwrite', action='store_true', help="Whether to overwrite the existing model")

args = parser.parse_args()

if not os.path.isdir('results_nobel'):
    os.mkdir('results_nobel')

model_name = args.mname

if os.path.isdir('./results_nobel/'+model_name):
    if args.overwrite:
        try:
            os.remove('./results_nobel/'+model_name+'/history.pkl')
            os.remove('./results_nobel/'+model_name+'/training_plot.png')
        except:
            print("==> Overwriting model...")
    else:
        error_msg = 'Model with name : '+model_name+' already exist!\nPlease enter a differenet model name.'
        raise ValueError(error_msg)
else:
    os.mkdir('./results_nobel/'+model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=(30,60)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


img, label = next(iter(trainloader))
print('Input Image Size: ',img.size())


# Selecting the Model
print('==> Building model..')

if args.model=='ResNet10':
    net = ResNet10()
elif args.model=='ResNet14':
    net = ResNet14()
elif args.model=='ResNet14_v2':
    net = ResNet14_v2()
net = net.to(device)


## Checking the number of trainable parameters
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print()
print("#"*50)
print("Number of Parameters to train: ", pytorch_total_params)
print("#"*50)
print()


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

## Selecting the optimizer
if args.optimz == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
elif args.optimz == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimz == 'rmsprop':
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum = 0.9, weight_decay=args.wd)
elif args.optimz == 'adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=args.wd)

## Perform Cosine Annealing
if args.do_annealing:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    avg_train_loss = train_loss/total
    avg_train_acc = 100.*correct/total
    return avg_train_loss, avg_train_acc


def test(epoch):
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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+model_name+'_ckpt.pth')
        best_acc = acc
    
    avg_val_loss = test_loss/total
    avg_val_acc = 100.*correct/total
    return avg_val_loss, avg_val_acc


history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

for epoch in range(start_epoch, start_epoch+args.epochs):    
    t_loss, t_acc = train(epoch)
    v_loss, v_acc = test(epoch)
    if args.do_annealing:
        scheduler.step()

    history['train_loss'].append(t_loss)
    history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss)
    history['val_acc'].append(v_acc)

### Saving History
hist_file = './results_nobel/' + model_name + '/history.pkl'
with open(hist_file, 'wb') as f:
    pickle.dump(history, f)

### Saving Training Plot
plot_file = './results_nobel/' + model_name + '/training_plot.png'
plot_title = 'Parameters: '+str(pytorch_total_params)+' | lr :'+str(args.lr)
save_plot_over_training(history, plot_title, plot_file)


print('\nTraining Complete !')
print('Best Accuracy: ',round(best_acc,4),'%')
print('Best Model saved in /checkpoint/'+model_name+'_ckpt.pth')
print('History and Plots saved in /results_nobel/'+model_name)