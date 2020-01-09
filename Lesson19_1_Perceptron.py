import torch
import torch.nn.functional as F

def test_perceptron():
    x = torch.rand(1,10)
    w = torch.rand(1,10, requires_grad=True)
    o = torch.sigmoid(x@w.t())
    print(o)
    loss = F.mse_loss(torch.ones(1,1), o)
    print(loss)
    print(w.grad)
    loss.backward()
    print(w.grad)
    o = torch.sigmoid(x@w.t())
    loss = F.mse_loss(torch.ones(1,1), o)
    print(loss)
    w.grad.zero_()
    print(w.grad)
    loss.backward()
    print(w.grad)

if __name__ == '__main__':
    test_perceptron()