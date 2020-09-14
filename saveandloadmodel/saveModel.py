import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Linear(1,1)
        self.layer.weight = nn.Parameter(torch.FloatTensor([[10]]))
        self.layer.bias = nn.Parameter(torch.FloatTensor([1]))

    def forward(self, x):
        y = self.layer(x)
        return y



x = torch.FloatTensor([[1]])
net = Net()
out = net(x)
print(out)

## 保存Net的参数值
# Net的参数值存储在其state_dict(状态字典)属性中
#我们查看一下net的state_dict包含哪些参数
print(net.state_dict())

#保存的函数是torch.save()，参数是我们需要保存的dict和存储路径
torch.save(obj=net.state_dict(), f="models/net.pth")

#加载Net参数值并用于新的模型
#最后一个步骤就是从pth文件中重新获取Net参数值,并把参数值装载到新定义的Model对象中。
#这里我们重新定义一个结构和Net类相同的类 Model， 区别仅仅是Model参数初始值和Net不同。

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Linear(1,1)
        self.layer.weight = nn.Parameter(torch.FloatTensor([[0]]))
        self.layer.bias = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, x):
        out = self.layer(x)
        return out

model = Model()
print(model.state_dict())

#现在，我们将model对象的参数值设置为net.pth中的值，需要使用model.load_state_dict() 函数重置model的参数值为"torch.load(models/net.pth)"
#中的参数值
model.load_state_dict(torch.load("models/net.pth"))
print(model.state_dict())

#优化器与epoch的保存
net = Net()
Adam = optim.Adam(params=net.parameters(), lr=0.001, betas=(0.5, 0.999))
epoch = 96

all_states = {"net":net.state_dict(), "Adam":Adam.state_dict(), "epoch": epoch}
torch.save(obj=all_states, f="models/all_states.pth")

reload_states = torch.load("models/all_states.pth")
print(reload_states)

