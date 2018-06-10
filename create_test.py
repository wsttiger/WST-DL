import numpy as np
import torch

torch.manual_seed(12345)
# number of input channels
ncin = 1
# number of output channels
ncout = 3
# dimensions of the kernel
fH = 3
fW = 3
# strides
s = 2
# paddings
p = 1
# total size
sz = ncin*ncout*fH*fW
# create convolution and convert into numpy array (linear)
Cv = torch.nn.Conv2d(ncin,ncout,(fH,fW),stride=s,padding=p,bias=False)
Carray = np.array(Cv.weight.data).reshape(sz)

fileConv2d = open('conv2d.dat', 'w')
print('%d   %d   %d   %d   %d   %d\n' % (ncin,ncout,fH,fW,s,p), end='', file=fileConv2d)
for i in range(0,sz):
  print('%15.10e ' % Carray[i], end='', file=fileConv2d)
print('', file=fileConv2d)

# batch size
N = 1
# image dimensions
nH = 6
nW = 6
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
print(Y)
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
