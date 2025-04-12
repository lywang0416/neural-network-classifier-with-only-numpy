# neural-network-classifier-with-only-numpy


本项目使用 **纯 NumPy 实现三层神经网络**，在 **CIFAR-10 图像分类任务**上进行训练和评估，**不依赖任何深度学习框架（如 PyTorch 或 TensorFlow）**。

---

##  项目结构

```bash
neural-network-classifier-with-only-numpy/
├── config.yaml                    # 训练超参配置
├── hyperparameter_search.py       # 搜索超参数时使用的函数
├── README.md  
├── run.py                         # 入口脚本
├── test.py.                       # 测试
├── train.py                       # 训练
└── utils.py                             
```

## 启动指令

```bash
python run.py --config config.yaml --search_mode learning_rate

python run.py --config config.yaml --search_mode reg_strength

python run.py --config config.yaml --search_mode hidden_size

python run.py --config config.yaml --search_mode all
```
分别为搜索学习率、正则化强度、隐藏层大小和所有超参数组合的指令。注意，在单独搜索某一个超参数时，config.yaml中不需要搜索的超参只能有一个值。
