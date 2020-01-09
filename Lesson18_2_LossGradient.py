import torch
import numpy as np
import torch.nn.functional as F

def test_autograd_loss():
    print("Enter test_autograd_loss")
    x = torch.ones(5)
    w = torch.tensor(2, dtype=torch.float).requires_grad_()
    mse = F.mse_loss(torch.ones(5), x*w)
    print(x, w, mse)
    grad = torch.autograd.grad(mse, w)
    print(grad)
    print("Exit test_autograd_loss")

def test_loss_backward():
    print("<<<<<<<<Enter test_loss_backward>>>>>>>>")
    x = torch.ones(2)
    w = torch.full([2],2, requires_grad=True)
    print(x, x.shape)
    print(w, w.shape)
    mse = F.mse_loss(torch.ones(2), x*w)
    print(mse)
    mse.backward()
    print(w.grad)
    mse = F.mse_loss(torch.ones(2), x*w)
    w.grad.zero_()
    print(mse)
    mse.backward()
    print(w.grad)
    print("<<<<<<<<Exit test_loss_backward>>>>>>>>")

def test_softmax():
    print("<<<<<<<<Enter test_softmax>>>>>>>>")
    a = torch.rand(3)
    a.requires_grad_()
    p = F.softmax(a,dim=0)
    print(a, p)
    p[0].backward(retain_graph=True)
    print(a.grad)
    a.grad.zero_()
    p[1].backward(retain_graph=True)
    print(a.grad)
    a.grad.zero_()
    p[2].backward()
    print(a.grad)
    mp = p.reshape(-1, 1)
    dv = -mp@mp.t()
    dv = dv - torch.diag(dv.diag())
    dv += torch.diag(p*(1.0-p))
    print(dv)
    print("<<<<<<<<Exit test_softmax>>>>>>>>")

if __name__ == "__main__":
    test_autograd_loss()
    test_loss_backward()
    test_softmax()

    npar = np.random.rand(4,4)
    npdi = np.diag(np.diag(npar))
    print(npar - npdi , sep='\n')