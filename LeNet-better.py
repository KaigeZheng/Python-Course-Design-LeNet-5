import torch
from torch import nn, optim
import time
import matplotlib.pyplot as plt
from MODULE import DataloadFasionMNIST as dl

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    
net = torch.nn.Sequential(Reshape(), 
                          nn.Conv2d(1, 6, kernel_size=5,padding=2), nn.Sigmoid(),nn.AvgPool2d(kernel_size=2, stride=2),
                          nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),nn.AvgPool2d(kernel_size=2, stride=2), 
                          nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                          nn.Linear(120, 84), nn.Sigmoid(), 
                          nn.Linear(84, 10))

batch_size = 256
train_iter, test_iter = dl.load_dataset(batch_size)

def train_LeNet(net, train_iter, test_iter, num_epochs, lr, device='cuda'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    
    print('training on', device)
    net.to(device)
    
    # 优化器 -> Adam
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 损失函数 -> 交叉熵
    loss = nn.CrossEntropyLoss()
    
    # 记录每个epoch的损失和准确率
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 开始迭代
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc, num_examples = 0.0, 0.0, 0
        start_time = time.time()
        
        # 训练阶段
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
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
        scheduler.step()
    
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
    save_path = "./LeNet-better.png"
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    plt.show()

# baseline
epochs = 100
lr = 0.01
train_LeNet(net, train_iter, test_iter, epochs, lr)