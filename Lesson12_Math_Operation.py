"""
torch.equal(input, other) → bool
True if two tensors have the same size and elements, False otherwise.

torch.mm(input, mat2, out=None) → Tensor
Performs a matrix multiplication of the matrices input and mat2.
If input is a (n x m)(n×m) tensor, mat2 is a (m x p)(m×p) tensor, out will be a (n x p)(n×p) tensor.
This function does not broadcast. For broadcasting matrix products, see torch.matmul().

torch.matmul(input, other, out=None) → Tensor
Matrix product of two tensors.
The behavior depends on the dimensionality of the tensors as follows:
If both tensors are 1-dimensional, the dot product (scalar) is returned.
If both arguments are 2-dimensional, the matrix-matrix product is returned.
If the first argument is 1-dimensional and the second argument is 2-dimensional,
    a 1 is prepended to its dimension for the purpose of the matrix multiply.
    After the matrix multiply, the prepended dimension is removed.
If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2),
    then a batched matrix multiply is returned. If the first argument is 1-dimensional,
     a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.
      If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix
       multiple and removed after. The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable).
        For example, if input is a (j×1×n×m) tensor and other is a (k×m×p) tensor, out will be an (j×k×n×p) tensor.
torch.rsqrt(input, out=None) → Tensor
Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
torch.clamp(input, min, max, out=None) → Tensor
Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
"""

import torch


# Test +-x/
def test_basic(a, b):
    print("Enter test_basic")
    print(a, "\n", b, a.type())
    print(a.shape, b.shape)
    print(a + b)
    print("Test +:", torch.equal(a + b, torch.add(a, b)))
    print("Test -:", torch.equal(a - b, torch.sub(a, b)))
    print("Test x:", torch.equal(a * b, torch.mul(a, b)))
    print("Test /:", torch.equal(a / b, torch.div(a, b)))
    print(torch.all(torch.eq(a / b, torch.div(a, b))))
    print("Exit test_basic")


def test_matmul(a, b):
    print("Enter test_matmul")
    print("a: ", a.shape, "\n", a)
    print("b: ", b.shape, "\n", b)
    print("matmul: ", "\n", torch.matmul(a, b))
    print("matmul: ", "\n", torch.matmul(b, a.t()))
    print("@: ", "\n", a @ b)
    b = b.view(a.shape[-1], -1)
    print("b: ", b.shape, "\n", b)
    print("mm: ", "\n", torch.mm(a, b))

    a = torch.rand(4, 3, 28, 64)
    b = torch.rand(4, 3, 64, 2)
    print("matmul:", torch.matmul(a, b).shape)

    print("Exit test_matmul")


def test_power():
    print("Enter test_power")
    a = torch.full(size=(2, 2), fill_value=3)
    aa = torch.pow(input=a, exponent=2)
    print(a, "\n", aa)
    print("test power **:", torch.equal(torch.pow(a, 5), a ** 5))
    print("test power **:", torch.equal(a, aa ** 0.5))
    print("rsqrt:", torch.rsqrt(aa))
    print("Exit test_power")


def test_explog():
    print("Enter test_explog")
    a = torch.ones(size=(2, 2), dtype=torch.float)
    b = a.exp()
    print(b)
    print("test log:", a.equal(torch.log(b)))
    print("Exit test_explog")


def test_appro():
    print("Enter test_appro")
    a = torch.tensor([-3.5, -3.1415, -3., 0.0, 3., 3.1415, 3.5])
    print("orig:", a)
    print("floor: ", torch.floor(a))
    print("ceil: ", torch.ceil(a))
    print("trunc: ", torch.trunc(a))
    print("frac: ", torch.frac(a))
    print("round: ", torch.round(a))
    print("Exit test_appro")


def test_clamp():
    torch.manual_seed(0)
    print("Enter test_clamp")
    grad = torch.rand(2, 3) * 15
    print(grad)
    print("orig:", grad)
    print("sort:", grad.flatten().sort())
    print("stat:", grad.max(), grad.min(), grad.mean(), grad.median())
    print("clamp:", grad.clamp(10))
    print("clamp0:", grad.clamp(0, 10))
    print("Exit test_clamp")


if __name__ == '__main__':
    a = torch.arange(start=0, end=12, dtype=torch.float).view(3, 4)
    b = torch.arange(start=101, end=105, dtype=torch.float)
    test_basic(a, b)

    test_matmul(a, b)
    test_power()
    test_explog()
    test_appro()
    test_clamp()
