"""
torch.prod()
torch.prod(input, dtype=None) → Tensor
Returns the product of all elements in the input tensor.

torch.argmax(input) → LongTensor
Returns the indices of the maximum value of all elements in the input tensor.
This is the second value returned by torch.max().

torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
Returns the k largest elements of the given input tensor along a given dimension.
If dim is not given, the last dimension of the input is chosen.
If largest is False then the k smallest elements are returned.
A namedtuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor.
The boolean option sorted if True, will make sure that the returned k elements are themselves sorted

torch.kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)
Returns a namedtuple (values, indices) where values is the k th smallest element of each row of the input tensor in the
 given dimension dim. And indices is the index location of each element found.
"""
import torch

def test_normp():
    print("Enter test_normp")
    a = torch.full(size=(12,), fill_value=1)
    b = a.view(3,4)
    c = a.view(3,2,2)
    print(a, b, c, sep='\n')
    for p in range(4):
        print(p, a.norm(p), b.norm(p), c.norm(p))

    print("Exit test_normp")

def test_stat():
    print("Enter test_stat")
    a = torch.arange(start=1, end=9, dtype=torch.float).view(2,4)
    print(a)
    print("sum:", a.sum())
    print("torchmax:", torch.max(a))
    print("torchmin:", torch.min(a))
    print("argmax:", a.argmax(dim=1, keepdim=True))
    print("argmin:", a.argmin())
    print("max:", a.max())
    print("min:", a.min())
    print("mean:", a.mean())
    print("media:", a.median())
    print("prod:", a.prod())
    print("Exit test_stat")


def test_topk():
    print("Enter test_topk")
    ind = torch.randperm(8)
    a = torch.arange(start=8, end=0, step=-1, dtype=torch.float)[ind].view(2,4)
    print(a)
    print(a.topk(k=3, dim=1, largest=False, sorted=True))

    print(a.kthvalue(k=3))
    print("Exit test_topk")

if __name__ == "__main__":
    test_normp()
    test_stat()
    test_topk()