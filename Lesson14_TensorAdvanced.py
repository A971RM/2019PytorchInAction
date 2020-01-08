"""
torch.where()
torch.where(condition, x, y) → Tensor
Return a tensor of elements selected from either x or y, depending on condition.

torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor
Gathers values along an axis specified by dim.

expand(*sizes) → Tensor
Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
"""
import torch

def test_advanced():
    print("Enter test_advanced")
    x = torch.randn(3, 2)
    y = torch.ones_like(x)
    print("x:", x, x.type(), "\n", "y:", y, y.type())
    print(torch.where(x > 0, x, y))

    prob = torch.rand(4, 10)
    idx = prob.topk(k=3, dim=1, sorted=True)
    print(prob, idx)
    lable = torch.arange(0, 10).expand_as(prob)
    print(lable, lable.shape, "\n", lable)
    print(lable.gather(dim=1, index=idx.indices))

    print("Exit test_advanced")


if __name__ == "__main__":
    test_advanced()
