import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import itertools
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt

initial_lr = 1e-4


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass


net_1 = model()

# optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
# scheduler_1 = CosineAnnealingWarmRestarts(optimizer_1, T_0=5, T_mult=2)

optimizer_1 = torch.optim.AdamW(net_1.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4,
                                  amsgrad=False)
# schedular_r = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_1, T_0=5, T_mult=2)
#
schedular_r = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_1, T_0=5, T_mult=2)
scheduler = GradualWarmupScheduler(optimizer_1, multiplier=10, total_epoch=10, after_scheduler=schedular_r)

print("初始化的学习率：", optimizer_1.defaults['lr'])

lr_list = []  # 把使用过的lr都保存下来，之后画出它的变化

for epoch in range(1, 250):
    # train

    optimizer_1.zero_grad()
    optimizer_1.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    lr_list.append(optimizer_1.param_groups[0]['lr'])
    scheduler.step()

# 画出lr的变化
plt.plot(list(range(1, 250)), lr_list)
plt.xlabel("epoch")
plt.ylabel("lr")
plt.title("learning rate's curve changes as epoch goes on!")
plt.show()