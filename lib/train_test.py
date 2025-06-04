'''
这个python文件，定义了训练和测试函数。用于训练神经网络模型。
'''

import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
# 定义训练函数

def train(dataloader, model, loss_fn, batch_size, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % batch_size == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# train multi channel data, and align the channel dimension to 4
def train_multi_channel(dataloader, model, loss_fn, batch_size, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        X_expanded = X.expand(25, 4, 28, 28).clone()

        # Compute prediction error
        pred = model(X_expanded)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % batch_size == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def train_multi_channel_32(dataloader, model, loss_fn, batch_size, optimizer, device):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         X_expanded = X.expand(25, 4, 32, 32).clone()

#         # Compute prediction error
#         pred = model(X_expanded)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % batch_size == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train_arr(dataloader, model, loss_fn, batch_size, optimizer, device, loss_list):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(X.dtype)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # save loss data
        # loss_list.append(loss)

        if batch % batch_size == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            loss_list.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print("size:", size)
    print("num_batches:", num_batches)
    model.eval()
    y_true, y_pred = [], []
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            predicted_labels = pred.argmax(1)
            correct += (predicted_labels == y).type(torch.float).sum().item()


            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

    test_loss /= num_batches
    correct /= size
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test Error: \n Accuracy: {(accuracy*100):.1f}%, Avg loss: {test_loss:.4f}, Precision: {(precision*100):.2f}%, Recall: {(recall*100):.2f}%, F1-Score: {(f1*100):.2f}%, \n")
    return accuracy, test_loss

# test multi channel data, and align the channel dimension to 4
def test_multi_channel(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # align channel dimension to 4
            # Repeat the input tensor along the channel dimension
            X_repeated = torch.cat([X] * 4, dim=1)
            pred = model(X_repeated)
            print(pred.shape)
            print(y.max())
            loss = loss_fn(pred, y).item()
            test_loss += loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# test multi channel data, and align the channel dimension to 4
# all: calculate accuracy, precision, recall, f1-score
def test_multi_channel_all(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    y_true, y_pred = [], []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X_repeated = torch.cat([X] * 4, dim=1)

            pred = model(X_repeated)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            predicted_labels = pred.argmax(1)
            correct += (predicted_labels == y).type(torch.float).sum().item()


            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

    test_loss /= num_batches
    correct /= size
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test Error: \n Accuracy: {(accuracy*100):.1f}%, Avg loss: {test_loss:.4f}, Precision: {(precision*100):.2f}%, Recall: {(recall*100):.2f}%, F1-Score: {(f1*100):.2f}%, \n")
    return accuracy, test_loss

# def test_multi_channel_32(dataloader, model, loss_fn, device):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     print("size:", size)
#     print("num_batches:", num_batches)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             X_repeated = torch.cat([X] * 4, dim=1)
#             pred = model(X_repeated)
#             loss = loss_fn(pred, y).item()
#             test_loss += loss
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# test & save accuracy data & avg_loss data
def test_arr(dataloader, model, loss_fn, device, acc_list, test_loss_list):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print("size:", size)
    print("num_batches:", num_batches)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    acc_list.append(correct) # save accuracy data
    test_loss_list.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

import numpy as np
def generate_normalized_matrix(m, n):
    """生成一个 m×n 的二维数组，元素非负且平方和为1"""
    vec = np.abs(np.random.randn(m, n))  # 生成正态分布随机数并取非负值
    norm = np.linalg.norm(vec)           # 计算L2范数（平方和开根号）
    vec /= norm                          # 归一化
    return vec

def generate_3d_array(k, m, n):
    """生成形状为 (k, m, n) 的三维数组，每层二维数组满足条件"""
    return np.array([generate_normalized_matrix(m, n) for _ in range(k)])


def print_2d_arr(array):
    """打印二维数组"""
    for i in range(0, len(array)):
        print(f"Model[{i + 1}]: ", end="")
        for j in range(0, len(array[i])):
            print(f"{array[i][j]:.4f}", end=", ")
        print()
    print()