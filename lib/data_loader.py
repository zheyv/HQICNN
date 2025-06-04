'''
负责数据集的加载和处理。同时打印采样后的信息。
'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import matplotlib.pyplot as plt
import numpy as np

def sample_balanced_classes(dataset, num_samples_per_class, label_list):
    '''对数据集进行平均采样。

    将dataset数据集，对label_list内的类别标签采样num_samples_per_class个样本，返回数据集的一个子集。 
    
    Keyword arguments:
    dataset                --- 数据集对象
    num_samples_per_class  --- 每个类别采样的样本数量
    label_list             --- 将label转化为数组，取范围内的类别标签。
    '''
    indices_per_class = {i: [] for i in label_list}
    for idx, (_, label) in enumerate(dataset):
        if label in indices_per_class and len(indices_per_class[label]) < num_samples_per_class:
            indices_per_class[label].append(idx)
    sampled_indices = [idx for indices in indices_per_class.values() for idx in indices]
    return Subset(dataset, sampled_indices)

# 可视化数据
def imshow(img):
    '''可视化灰度图像数据

    '''
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis('off')

def print_samples(images, labels, num_samples=5):
    '''打印样本
    '''
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        imshow(images[i])
        plt.title(f"Label: {labels[i].item()}")
    plt.show()

# 获取MNIST数据集的dataloader, 并可视化数据
def getDataloader(train_dataset, test_dataset, label_range, train_size=1200, test_size=300, batch_size=50, shuffle=True):
    # 平衡采样
    train_dataset = sample_balanced_classes(train_dataset, train_size, label_range)
    test_dataset = sample_balanced_classes(test_dataset, test_size, label_range)

    # 数据读取类，一次读取batch_size个样本
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    # 查看数据集的形状
    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]:+--------------------------------------------------------------- {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # 获取前5个训练和测试样本
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    train_images, train_labels = next(train_iter)
    test_images, test_labels = next(test_iter)

    print("train_images's shape is : ", train_images.shape)
    print("train_labels's shape is : ", train_labels.shape)

    # 绘制训练样本
    print("Training Samples:")
    print_samples(train_images, train_labels);

    # 绘制测试样本
    print("Testing Samples:")
    print_samples(test_images, test_labels);

    return train_loader, test_loader

# add luminance channel for CIFAR-10
def add_luminance_channel(img):
    luminance = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    luminance = luminance.unsqueeze(0)  # 增加通道维度
    return torch.cat([img, luminance], dim=0)

# 
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        noise = torch.randn(tensor.size(), dtype=torch.float32) * self.std + self.mean
        return torch.clamp(tensor + noise, 0, 1)  # 保证像素值仍然在 [0,1]