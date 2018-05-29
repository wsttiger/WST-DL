import numpy as np
import torch

ncin = 3
ncout = 5
fH = 3
fW = 3
sz = ncin*ncout*fH*fW
C = torch.nn.Conv2d(ncin,ncout,(fH,fW))
Carray = np.array(C.weight.data).reshape(sz)

fileConv2d = open('conv2d.dat', 'w')
print('%d    %d    %d    %d\n' % (ncin,ncout,fH,fW), end='', file=fileConv2d)
for i in range(0,sz):
  print('%15.10e    ' % Carray[i], end='', file=fileConv2d)
print('', file=fileConv2d)
