import torch
import torch.nn.functional as F

def test_lottery():
    a = torch.full(size=[4], fill_value=1/4)
    print(a)
    entropy = a * torch.log2(a)
    entropy = -entropy.sum()
    print(entropy)

    a = torch.tensor([0.1, 0.1, 0.1, 0.7])
    print(a)
    entropy = a * torch.log2(a)
    entropy = -entropy.sum()
    print(entropy)

    a = torch.tensor([0.01, 0.01, 0.01, 0.97])
    print(a)
    entropy = a * torch.log2(a)
    entropy = -entropy.sum()
    print(entropy)

    a = torch.tensor([0.00001, 0.00001, 0.00001, 0.99997])
    print(a)
    entropy = a * torch.log2(a)
    entropy = -entropy.sum()
    print(entropy)

    a = torch.tensor([1e-10, 1e-10, 1e-10, 1-3e-10])
    print(a)
    entropy = a * torch.log2(a)
    entropy = -entropy.sum()
    print(entropy)


def test_stability():
    x = torch.rand(1, 784)
    w = torch.rand(10, 784)
    logits = x@w.t()
    print(logits)
    pred = F.softmax(logits, dim=1)
    pred_log = torch.log(pred)
    print(pred)
    print(pred_log)
    print(F.cross_entropy(logits, torch.tensor([3])))
    print(F.nll_loss(pred_log, torch.tensor([3])))

if __name__ == "__main__":
    test_lottery()
    test_stability()