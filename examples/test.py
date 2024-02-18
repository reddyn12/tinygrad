from tinygrad import Tensor, dtypes,nn
from einops import rearrange, repeat, reduce
import numpy as np
import torch

t1 = torch.arange(64)
t2 = Tensor.arange(64)
p1 = torch.nn.Embedding(64,512)
p2 = nn.Embedding(64,512)
print(p1(t1))
print(p2(t2).numpy())

# r = Tensor.ones(16,64)
# r= r.one_hot(9).float()
# print(r.numpy())
# print(r.shape)
# for s in r.shape:
#     print(s, type(s))


# temp = (1.,2.,3.)
# print(temp)

# t1 = torch.ones(16, 64, 64, 64, 3)
# t2 = Tensor.ones(16, 64, 64, 64, 3)
# a = [t2, t2]
# o1 = torch.cat([t1,t1], dim=0)
# o2 = Tensor.cat(*a, dim=0)

# print(o1.shape)
# print(o2.shape)

# dim = 10
# t = np.arange(1000).reshape(10,10,10).astype(np.float32)
# B=5
# tNew = rearrange(t, "B L (C H W) -> (B L) C H W", C=B, H=1)
# # tNew = reduce(t, "B L C H W -> B L", "sum")
# # tNew = repeat(t, "B G N L -> B (G H) N L", H=dim // t.shape[1])
# # # tNew = rearrange(t, "b d l -> (b l) d")
# tTiny = Tensor.arange(1000).reshape(10,10,10).cast(dtypes.float32)
# # tTinyNew = tTiny.repeat((1,1,1,B*1))#.repeat((1,1,1,B,1))
# tTinyNew = tTiny.reshape(10*10,B,1,10//5//1)
# # tTinyNew = tTiny.repeat((B,1,1,1,1))
# # print(tTiny.shape, tTinyNew.shape)
# # tTinyNew = tTinyNew.reshape(tTinyNew.shape[0],tTinyNew.shape[1],
# #                                              tTinyNew.shape[2]*tTinyNew.shape[3]*tTinyNew.shape[4])
# # tTinyNew = tTiny.sum(-1).sum(-1).sum(-1)
# # tTinyNew = tTiny.repeat((1,dim//tTiny.shape[1],1,1))#.reshape(4,2)
# # # tTinyNew = tTiny.permute(0,2,1)#.reshape(4,2)
# print(t)
# print('BBBBBBBBBBBBBB')
# print(tNew)
# print(tNew.shape)
# # print(tTiny.numpy())
# # # print(tTiny.shape)

# print(tTinyNew.numpy())
# print(tTinyNew.shape)

# # print(tNew.shape, tTinyNew.shape)

# # t1 = torch.arange(16).float()
# # t2 = Tensor.arange(16).float()

# # v1 = t1.masked_fill(t1>10, -1e9)


# # print(t1.numpy())
# # print(v1.numpy())









# # t1 = torch.arange(16).reshape(4,4)+10
# # t2 = Tensor.arange(16).reshape(4,4)+10

# # v1k = t1.kthvalue(3).values
# # v1 = t1[3]
# # v2 = t2[3]

# # print(t1.numpy(), t2.numpy())
# # print(v1.numpy(), v2.numpy())
# # print(v1k.numpy())