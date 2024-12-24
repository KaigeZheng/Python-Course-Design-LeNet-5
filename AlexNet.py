import torch
from torch import nn, optim
import time
import matplotlib.pyplot as plt
from MODULE import DataloadFasionMNIST as dl

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

batch_size = 128
train_iter, test_iter = dl.load_dataset(batch_size, resize=224)

def train_AlexNet(net, train_iter, test_iter, num_epochs, lr, device='cpu'):
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    
    print('Training on', device)
    net.to(device)
    
    # 优化器 -> SGD
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    # 损失函数 -> 交叉熵
    loss_fn = nn.CrossEntropyLoss()
    
    # 记录每个epoch的损失和准确率
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 开始训练
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc, num_examples = 0.0, 0.0, 0
        start_time = time.time()
        
        # 训练阶段
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            l.backward()
            optimizer.step()
            
            train_loss += l.item() * X.size(0)
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()
            num_examples += X.size(0)
        
        train_loss /= num_examples
        train_acc /= num_examples
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 评估阶段
        net.eval()
        test_acc, num_test_examples = 0.0, 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                test_acc += (y_hat.argmax(dim=1) == y).sum().item()
                num_test_examples += X.size(0)
        test_acc /= num_test_examples
        test_accuracies.append(test_acc)
        
        end_time = time.time()
        
        print(f'Epoch {epoch + 1}: '
              f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}, '
              f'time {(end_time - start_time):.2f} sec')
    
    print('Training complete')
    
    # 绘制图表
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制训练损失
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    # 绘制训练准确率
    plt.plot(epochs, train_accuracies, 'go-', label='Train Accuracy')
    # 绘制测试准确率
    plt.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

train_AlexNet(net, train_iter, test_iter, epochs, lr)