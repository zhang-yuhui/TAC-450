# November 6, 2023
# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from line_profiler import profile

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer:
        # - Input channels: 1 (e.g., grayscale images)
        # - Output channels: 32
        # - Kernel size: 3x3
        # - Stride: 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        
        # Second convolutional layer:
        # - Input channels: 32 (from conv1)
        # - Output channels: 64
        # - Kernel size: 3x3
        # - Stride: 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        
        # First dropout layer:
        # - Dropout probability: 50%
        # - Further regularizes the model during training
        self.dropout1 = nn.Dropout(p=0.5)
        
        # First fully connected (linear) layer:
        # - Input features: 7x7x64 = 3136
        #   (Assuming input image size is 28x28, after two conv layers and pooling)
        # - Output features: 1024
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=1024)
        
        # Second fully connected (linear) layer:
        # - Input features: 1024
        # - Output features: 10 (e.g., number of classes for classification)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # Pass input through the first convolutional layer
        x = self.conv1(x)
        
        # Apply ReLU activation function
        x = F.relu(x)

        # Max Pool
        x = F.max_pool2d(x,kernel_size=2, stride=2)
        
        # Pass the result through the second convolutional layer
        x = self.conv2(x)
        
        # Apply ReLU activation function
        x = F.relu(x)

        # Max Poll
        x = F.max_pool2d(x,kernel_size=2, stride=2)
        
        # Flatten the tensor starting from the first dimension (excluding batch size)
        # This prepares the data for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Pass through the first fully connected layer
        x = self.fc1(x)

        # Apply ReLU activation function
        x = F.relu(x)

        # Apply the dropout layer(p=0.5)
        x = self.dropout1(x)
        
        x = self.fc2(x)

        # Apply log softmax activation function to obtain log probabilities for each class
        output = F.log_softmax(x, dim=1)
        
        return output

@profile
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data/', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
