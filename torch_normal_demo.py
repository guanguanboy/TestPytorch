from torch.distributions.normal import Normal
import torch

#q_dist = Normal(q_mean, q_stddev)

m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
print(m.sample()) #直接在定义的正太分布上采样

print(m.rsample()) #https://blog.csdn.net/geter_cs/article/details/90752582
#
"""
先对标准正太分布N ( 0 , 1 ) N(0,1)N(0,1)进行采样，然后输出：
m e a n + s t d × 采 样 值 mean +  std * 采样值
mean+std×采样值
"""