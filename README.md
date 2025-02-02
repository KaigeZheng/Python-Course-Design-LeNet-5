# Optimization-on-LeNet-and-AlexNet

## 运行环境

### 硬件

+ `CPU`:Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz

+ `GPU`:Nvidia Tesla V100-PCIE 32GB

### 软件

+ `OS`:(WSL2)Linux Ubuntu 22.04.2 LTS

+ `Compiler`:Python 3.11

+ `cuda`:12.2

## 文件说明

```
main/
│  AlexNet-better.py
│  AlexNet.ipynb
│  AlexNet.py
│  DataloadFasionMNIST.ipynb
│  LeNet-better.py
│  LeNet.ipynb
│  LeNet.py
│  README.md
│
├─logfile
│      log-AlexNet-baseline
│      log-AlexNet-better
│      log-LeNet-baseline
│      log-LeNet-better
│
├─MODULE
│      DataloadFasionMNIST.py
│
└─outputfile
        AlexNet-better.png
        AlexNet.png
        LeNet-better.png
        LeNet.png
```

## 运行方法（Linux）

在线运行

```bash
python AlexNet-better.py > ./logfile/log-AlexNet-better
```

离线运行

```bash
nohup python AlexNet-better.py > ./logfile/log-AlexNet-better &
```
