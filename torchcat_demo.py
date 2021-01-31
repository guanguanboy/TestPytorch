import torch
A = torch.ones(2, 3)
print(A)

B = 2 * torch.ones(4, 3)
print(B)

"""
C=torch.cat((A,B),0)就表示按维数0（行）拼接A和B，
也就是竖着拼接，A上B下。此时需要注意：列数必须一致，即维数1数值要相同，
这里都是3列，方能列对齐。拼接后的C的第0维是两个维数0数值和，即2+4=6.
"""
C = torch.cat((A, B), 0)#按维数0（行）拼接,沿着第0个axis拼接，行数增加
print(C)
print('C = torch.cat((A, B), 0):', C.size())

F = torch.cat([A, B], 0)#按维数0（行）拼接
print('F = torch.cat([A, B], 0:',F.shape)

D = 2*torch.ones(2,4)

E = torch.cat([A, D], 1) ##按维数1（列）拼接，沿着第一个axis评价，列数增加
print(E.size())
print('E = torch.cat([A, D], 1):',E.shape)
