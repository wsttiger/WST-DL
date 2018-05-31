import numpy as np
import torch

# number of input channels
ncin = 3
# number of output channels
ncout = 5
# dimensions of the kernel
fH = 3
fW = 3
# strides
sH = 1
sW = 1
# paddings
pH = 0
pW = 0
# total size
sz = ncin*ncout*fH*fW
# create convolution and convert into numpy array (linear)
Cv = torch.nn.Conv2d(ncin,ncout,(fH,fW))
Carray = np.array(Cv.weight.data).reshape(sz)
# zero the bias of Cv
Cv.bias.data = torch.zeros_like(Cv.bias.data)

fileConv2d = open('conv2d.dat', 'w')
print('%d   %d   %d   %d   %d   %d   %d   %d\n' % (ncin,ncout,fH,fW,sH,sW,pH,pW), end='', file=fileConv2d)
for i in range(0,sz):
  print('%15.10e ' % Carray[i], end='', file=fileConv2d)
print('', file=fileConv2d)

# batch size
N = 2
# image dimensions
nH = 4
nW = 4
# total size
sz = N*ncin*nH*nW
# create torch array and convert to numpy (linear)
X = torch.randn(N,ncin,nH,nW)
Xarray = np.array(X).reshape(sz)
fileInput= open('input.dat', 'w')
print('%d    %d    %d    %d\n' % (N,ncin,nH,nW), end='', file=fileInput)
for i in range(0,sz):
  print('%15.10e ' % Xarray[i], end='', file=fileInput)
print('', file=fileInput)
# convert to autograd variable
X = torch.autograd.Variable(X)

# perform convolution in PyTorch
Y = Cv(X)
sz = Y.numel()
nH = Y.size(2)
nW = Y.size(3)
# convert to numpy array
Y = Y.detach().numpy().reshape(sz)
fileOutput= open('output.dat', 'w')
print('%d    %d    %d    %d\n' % (N,ncout,nH,nW), end='', file=fileOutput)
for i in range(0,sz):
  print('%15.10e ' % Y[i], end='', file=fileOutput)
print('', file=fileOutput)

