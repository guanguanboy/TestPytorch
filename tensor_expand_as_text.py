import torch
x = torch.tensor([[1], [2], [3]])
print('xsize:',x.size())
print('x:',x)

x_expand=x.expand(3,4)
print('x_expand:',x_expand)

x_expand=x.expand(-1,3)  # -1 means not changing the size of that dimension
print('x_expand:',x_expand)

x_expand_as=x.expand_as(x_expand)
print('x_expand_as:',x_expand_as)
