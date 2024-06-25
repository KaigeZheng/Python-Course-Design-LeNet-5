import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

dataset_path = '/home/kambri/DATA' 
batch_size = 256
workers = 4

'''返回FasionMNIST的文本标签'''
def get_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

'''展示数据集'''
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    # 设置图像尺寸
    figsize = (num_cols * scale, num_rows * scale) # 指定num_rows行num_cols列的子图网格
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.numpy() # 将张量转化为numpy数组
            if img.ndim == 3:
                img = img.transpose(1, 2, 0)
                # 映射灰度图
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        else:
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.axis('off')  # 隐藏坐标轴
        if titles:
            ax.set_title(titles[i])
    
    plt.tight_layout() # 调整多图，防止重叠
    plt.show()
    return axes

'''指定工作进程,可修改'''
def get_workers():
    return workers

'''下载数据集并加载到内存中'''
def load_dataset(batch_size, resize=None):  
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_workers()))

'''测试加载数据集,并输出数据集信息和展示图片,例如调用__test__(32, resize=64),当resize=None时分辨率为784=28*28'''
def __test__(batch_size, resize=28):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, transform=trans, download=True)
    print('The size of dataset(train) is {}. The size of dataset(test) is {}.'.format(len(mnist_train), len(mnist_test)))
    for X, y in mnist_train:
        print('Dataset(train) each element X: Shape:{}  Type:{}'.format(X.shape, X.dtype))
        break
    X, y = next(iter(data.DataLoader(mnist_train, batch_size)))
    show_images(X.reshape(batch_size, resize, resize), 2, batch_size//2, titles=get_labels(y))
