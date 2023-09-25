# snn


## 过程实验
1. transformer    AdamW(lr=0.001, weight_decay=0.009)    CosineAnnealingLR(eta_min=0, T_max=1000)    1000epoch  
    best_acc: 55%    
2. resnet20_m2    SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)    CosineAnnealingLR(eta_min=0, T_max=1000)  1000epoch
    best_acc: 76.2% 【parameters没有进行split_weights,应该是没有添加spike】 ×   
3. 
