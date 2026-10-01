# Homework Results

## 1. Brief Description of the Model Architecture

The model is a convolutional neural network used to classify 28 × 28 grayscale MNIST images into 10 digit classes. Overall, the model has 2 convolutional layers with pooling, and then flattens to connect to a MLP and a dropout layer before go through the final output layer for the logits.

## 2. Line-Profiler Results and Analysis

The training function was profiled for five epochs with a batch size of 64 on an NVIDIA V100 GPU. Both runs processed the same 4,690 training batches; the only experimental change was the data loader's `num_workers` setting.

| Profiled operation | 1 worker time (s) | 8 workers time (s) |1 worker % time | 8 workers % time |
|---|---:|---:|---:|---:|
| Complete training function | 70.788 | 39.782 | 100.0% | 100.0% |
| Iterate overbatches from `train_loader` | 43.607 | 19.378 | 61.6% | 48.7% |
| Data transfer to GPU | 0.697 | 0.906 | 1.0% | 2.3% |
| Clear gradients | 1.119 | 0.903 | 1.6% | 2.3% |
| Forward pass | 7.764 | 4.344 | 11.0% | 10.9% |
| Compute loss | 0.442 | 0.398 | 0.6% | 1.0% |
| Backward pass | 10.382 | 9.081 | 14.7% | 22.8% |
| Optimizer update | 6.703 | 4.657 | 9.5% | 11.7% |

Increasing the number of workers from 1 to 8 reduced the profiled training time by **more than 30 seconds**, corresponding to a **1.78× speedup**. The largest improvement was on the data-loader iteration line, whose time decreased from 43.607 seconds to 19.378 seconds, reducing more than half of the time. This shows that the one-worker run was strongly limited by input loading and preprocessing. With eight workers, several batches can be loaded and transformed in parallel, allowing data preparation to overlap more effectively with GPU computation.

## 3. Line-Profiler Output
```python
# Using 1 worker
Timer unit: 1e-06 s

Total time: 70.788 s
File: mnist_classify.py
Function: train at line 86

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    86                                           @profile
    87                                           def train(args, model, device, train_loader, optimizer, epoch):
    88         5       1714.3    342.9      0.0      model.train()
    89      4695   43606549.3   9287.9     61.6      for batch_idx, (data, target) in enumerate(train_loader):
    90      4690     696786.5    148.6      1.0          data, target = data.to(device), target.to(device)
    91      4690    1118875.0    238.6      1.6          optimizer.zero_grad()
    92      4690    7764073.7   1655.5     11.0          output = model(data)
    93      4690     442082.5     94.3      0.6          loss = F.nll_loss(output, target)
    94      4690   10381538.1   2213.5     14.7          loss.backward()
    95      4690    6702560.7   1429.1      9.5          optimizer.step()
    96      4690       9083.9      1.9      0.0          if batch_idx % args.log_interval == 0:
    97       940       8140.9      8.7      0.0              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    98       470      13034.3     27.7      0.0                  epoch, batch_idx * len(data), len(train_loader.dataset),
    99       470      42950.8     91.4      0.1                  100. * batch_idx / len(train_loader), loss.item()))
   100       470        638.0      1.4      0.0              if args.dry_run:
   101                                                           break

 70.79 seconds - mnist_classify.py:86 - train
```
```python
# Using 8 workers
Timer unit: 1e-06 s

Total time: 39.7821 s
File: mnist_classify.py
Function: train at line 86

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    86                                           @profile
    87                                           def train(args, model, device, train_loader, optimizer, epoch):
    88         5       1790.9    358.2      0.0      model.train()
    89      4695   19378315.3   4127.4     48.7      for batch_idx, (data, target) in enumerate(train_loader):
    90      4690     905807.2    193.1      2.3          data, target = data.to(device), target.to(device)
    91      4690     902823.5    192.5      2.3          optimizer.zero_grad()
    92      4690    4343539.4    926.1     10.9          output = model(data)
    93      4690     398137.2     84.9      1.0          loss = F.nll_loss(output, target)
    94      4690    9080957.8   1936.2     22.8          loss.backward()
    95      4690    4656852.6    992.9     11.7          optimizer.step()
    96      4690       4910.9      1.0      0.0          if batch_idx % args.log_interval == 0:
    97       940       4471.4      4.8      0.0              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    98       470       6681.0     14.2      0.0                  epoch, batch_idx * len(data), len(train_loader.dataset),
    99       470      97461.5    207.4      0.2                  100. * batch_idx / len(train_loader), loss.item()))
   100       470        362.3      0.8      0.0              if args.dry_run:
   101                                                           break

 39.78 seconds - mnist_classify.py:86 - train
 ```