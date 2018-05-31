import torch
import numpy as np
import scipy.signal as sig

X = torch.randn(1,1,4,4)
Xt = np.array(X).reshape((4,4))

Cv = torch.nn.Conv2d(1,1,(3,3))
C = np.array(Cv.weight.data).reshape((3,3))
B = np.array(Cv.bias.data).reshape(1)

print(C)
print(Xt)

Yv = Cv(X).detach().numpy().reshape((2,2))
Y = sig.convolve2d(np.flip(np.flip(C,0),1), Xt, 'valid')+B

print(Yv)
print(Y)
print(B)
#print(sig.convolve2d(C, Xt, 'valid'))
