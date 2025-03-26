# Install specific versions of PyTorch and torchvision (required for compatibility)
!pip3 install torch==1.5.0 torchvision==1.6.0

# Import core PyTorch and torchvision modules for model building, datasets, and training
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub  # Used for quantizing and dequantizing tensors

# Define a transform pipeline to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),                   # Converts image to tensor
    transforms.Normalize((0.5,), (0.5,))     # Normalize to [-1, 1] range
])

# Load MNIST dataset for training with transformations applied
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64,
                         shuffle=True, num_workers=16, pin_memory=True)

# Load MNIST test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64,
                        shuffle=False, num_workers=16, pin_memory=True)

# Utility class for tracking average values like loss and accuracy during training
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Resets all stats to zero"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates with a new value and sample size n"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns formatted string output"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# Compute Top-1 classification accuracy for a batch
def accuracy(output, target):
    """Computes the top-1 accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)  # Get index of max log-probability
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_one = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_one.mul_(100.0 / batch_size).item()

# Utility function to measure model file size (saved as .pt temporarily)
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

# Transfers weights from a regular model to a quantized model structure
def load_model(quantized_model, model):
    state_dict = model.state_dict()
    model = model.to('cpu')
    quantized_model.load_state_dict(state_dict)

# Fuses layers like Conv + ReLU, Linear + ReLU to optimize for quantization
def fuse_modules(model):
    torch.quantization.fuse_modules(model, [['conv1', 'relu1'],
                                            ['conv2', 'relu2'],
                                            ['fc1', 'relu3'],
                                            ['fc2', 'relu4']], inplace=True)

# ========================== 🔧 PERFORMANCE & FUNCTIONALITY ENHANCEMENTS ==========================

# 1. Use model.eval() before inference to disable dropout/batchnorm randomness
# Always evaluate models in inference mode to ensure stable predictions
# Example:
# model.eval()
# test(model, testloader)

# =================================================================================================

# 2. Cache datasets to memory using PersistentWorkers and prefetch_factor (if system has RAM)
# Speeds up data loading by reducing disk I/O
# DataLoader(..., persistent_workers=True, prefetch_factor=2)

# Example:
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4,
#                          pin_memory=True, persistent_workers=True, prefetch_factor=2)

# =================================================================================================

# 3. Replace view() with reshape() in forward pass to avoid future errors on non-contiguous tensors
# Safer and more flexible when handling quantized tensors or inputs coming from custom layers
# Use:
# x = x.reshape(x.shape[0], -1)
# Instead of:
# x = x.view(x.shape[0], -1)

# =================================================================================================

# 4. Reduce memory consumption using torch.cuda.amp (Automatic Mixed Precision)
# Available in PyTorch >= 1.6. Boosts training speed using FP16 while maintaining accuracy
# Usage:
# scaler = torch.cuda.amp.GradScaler()
# with torch.cuda.amp.autocast():
#     output = model(inputs)
#     loss = criterion(output, labels)

# =================================================================================================

# 5. Try other quantization backends for performance tuning (especially on ARM or Android)
# Example: QNNPACK, FBGEMM
# torch.backends.quantized.engine = 'qnnpack' or 'fbgemm'

# =================================================================================================

# 6. Add validation support during training loop to monitor overfitting
# Allows early stopping, learning rate scheduling, etc.
# Example structure:
# if epoch % 1 == 0:
#     val_acc = test(model, validation_loader)

# =================================================================================================

# 7. Log metrics to TensorBoard for visualization
# Helps track accuracy/loss/size improvements over time
# Setup:
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# writer.add_scalar('Loss/train', loss.item(), epoch)

# =================================================================================================

# 8. Replace CrossEntropyLoss with LabelSmoothingCrossEntropy for noisy data
# Helps regularize classification model by preventing overconfidence
# from torch.nn.functional import log_softmax
# def label_smoothing_loss(pred, target, epsilon=0.1):
#     n_class = pred.size(1)
#     one_hot = torch.zeros_like(pred).scatter(1, target.view(-1,1), 1)
#     one_hot = one_hot * (1 - epsilon) + (1 - one_hot) * epsilon / (n_class - 1)
#     return (-one_hot * log_softmax(pred, dim=1)).sum(dim=1).mean()

# =================================================================================================

# 9. Benchmark model inference time on CPU vs GPU
# Helps evaluate quantization speed-up in real-world scenarios
# import time
# start = time.time()
# _ = model(inputs)
# print(f"Inference Time: {time.time() - start:.3f}s")

# =================================================================================================

# 10. Test multiple qconfigs like 'qnnpack', 'fbgemm', or custom QConfig for optimized performance
# torch.quantization.get_default_qconfig('fbgemm')
# torch.quantization.get_default_qconfig('qnnpack')
# Can affect speed and size differently depending on deployment target

# =================================================================================================

# 11. Export quantized model to TorchScript for deployment
# scripted_model = torch.jit.script(quantized_model)
# scripted_model.save("model_quantized.pt")
# Enables use in C++ or Android-based production systems

# =================================================================================================

# 12. Measure model size with pickle protocol=4 for precise storage estimates (esp. on old systems)
# torch.save(model.state_dict(), "temp.p", _use_new_zipfile_serialization=False)

# =================================================================================================

# 13. Prune model weights before quantization (experimental for reducing model size further)
# Use torch.nn.utils.prune to remove unimportant weights
# import torch.nn.utils.prune as prune
# prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# =================================================================================================

# 14. Use learning rate scheduler to stabilize training
# Helps avoid plateauing or oscillating losses
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# scheduler.step()

# =================================================================================================

# 15. Reduce num_workers from 16 → 4 or 8 if training on laptops (helps avoid CPU thread contention)
# Especially useful in low-core machines or shared environments (e.g., Colab, Jupyter)

# =================================================================================================
