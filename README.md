# snn


## 过程实验
1. transformer    AdamW(lr=0.001, weight_decay=0.009)    CosineAnnealingLR(eta_min=0, T_max=1000)    1000epoch  
    best_acc: 55%    
2. resnet20_m2    SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)    CosineAnnealingLR(eta_min=0, T_max=1000)  1000epoch
    best_acc: 76.2% 【parameters没有进行split_weights,应该是没有添加spike】 ×   
3. resnet20_m2    SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)    CosineAnnealingLR(eta_min=0, T_max=1000)  1000epoch
    best_acc: 64.0% 【parameters没有进行split_weights,添加了spike】 训练在60个epoch时loss发生剧烈波动
4. resnet20_m2    AdamW(lr=0.001, weight_decay=0.009)   CosineAnnealingLR(eta_min=0, T_max=1000)    1000epoch
    best_acc: 69.06%    【parameters没有进行split_weights,添加了spike】
5. resnet20_m     SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)    CosineAnnealingLR(eta_min=0, T_max=num_epochs) 1000epoch
    best_acc: ...%  【将mem更新mem = mem * decay + x_in,更换成学习权重,并对权重经过一个tanh，每个LIFAct单独设立,不共享参数】
6. resnet20_m     SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)    CosineAnnealingLR(eta_min=0, T_max=num_epochs) 1000epoch
    best_acc: ...%  【将mem更新mem = mem * decay + x_in,更换成学习权重,并对权重经过一个tanh，每个LIFAct模块共享更新参数】
