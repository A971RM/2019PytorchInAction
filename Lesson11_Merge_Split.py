"""
https://pytorch.org/docs/stable/torch.html#torch.split

torch.cat() can be seen as an inverse operation for torch.split() and torch.chunk().

torch.cat(tensors, dim=0, out=None) → Tensor
Concatenates the given sequence of seq tensors in the given dimension.
All tensors must either have the same shape (except in the concatenating dimension) or be empty.

torch.split(tensor, split_size_or_sections, dim=0)[SOURCE]
Splits the tensor into chunks.
If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible).
Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.

torch.chunk(input, chunks, dim=0) → List of Tensors
Splits a tensor into a specific number of chunks.
Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.

torch.stack(tensors, dim=0, out=None) → Tensor
Concatenates sequence of tensors along a new dimension.
All tensors need to be of the same size.
"""

import sys
import torch


def test_cat():
    print("In test_cat",)
    # Statistics about scores

    # [class1-4, students, scores]
    class14 = torch.rand(size=(4, 32, 8))
    # [class5-9, students, scores]
    class59 = torch.rand(size=(5, 32, 8))
    print(class14.shape, class59.shape)

    class19 = torch.cat(tensors=(class14, class59), dim=0)
    print(class19.shape)

    a1 = torch.rand(size=[4,3,32,32])
    a2 = torch.rand(size=[5,3,32,32])
    a12=torch.cat(tensors=[a1, a2], dim=0)
    print(a12.shape)

    a2 = torch.rand(size=[4,1,32,32])
    a12=torch.cat(tensors=[a1, a2], dim=1)
    print(a12.shape)

    a1 = torch.rand(size=[4,3,16,32])
    a2 = torch.rand(size=[4,3,16,32])
    a12=torch.cat(tensors=[a1, a2], dim=2)
    print(a12.shape)
    print("Out test_cat",)


def test_stack():
    print("IN test_stack")
    print("Create new dim")
    a1 = torch.rand(size=[4,3,16,32])
    a2 = torch.rand(size=[4,3,16,32])
    a12cat = torch.cat(tensors=[a1, a2], dim=2)
    print(a12cat.shape)
    a12stack = torch.stack(tensors=[a1, a2, a2, a1, a1], dim=2)
    print(a12stack.shape)
    a1 = torch.rand(32, 8)
    a2 = torch.rand(32, 8)
    a12stack = torch.stack(tensors=[a1, a2], dim=0)
    print(a12stack.shape)
    a2 = torch.rand(30, 8)
    try:
        a12stack = torch.stack(tensors=[a1, a2], dim=0)
        print(a12stack.shape)
    except:
        print("Except.", sys.exc_info())

    a12cat = torch.cat(tensors=[a1, a2], dim=0)
    print(a1.shape, a2.shape, a12cat.shape)

    print("OUT test_stack")

def test_split():
    print("IN test_split")
    a = torch.rand(32, 8)
    b = torch.rand(32, 8)
    abstack = torch.stack(tensors=(a, b, a, b), dim=0)
    print(a.shape, b.shape, abstack.shape)
    aa, bb = abstack.split(split_size=3,dim=0)
    print(aa.shape, bb.shape)
    print("OUT test_split")


def test_chunk():
    print("IN test_chunk")
    a = torch.rand(32, 8)
    b = torch.rand(32, 8)
    abstack = torch.stack(tensors=(a, b, a, b,a,b,a), dim=0)
    print(a.shape, b.shape, abstack.shape)
    aa, bb, cc, dd = abstack.chunk(chunks=4,dim=0)
    print(aa.shape, bb.shape)
    print("OUT test_chunk")

if __name__ == "__main__":
    test_cat()
    test_stack()
    test_split()
    test_chunk()

    a = torch.rand(32, 8)
    b = torch.rand(32, 8)
    abstack = torch.stack(tensors=(a, b, a, b,a,b), dim=0)
    print("--------------")
    print(abstack.shape)
    itr = 0
    for st in abstack.split(split_size=2, dim=0):
        print(itr, st.shape)
        itr += 1
    print(len(abstack.split(split_size=5, dim=0)))
    print(len(abstack.chunk(chunks=5, dim=0)))
    print("This is for chunk")
    itr = 0
    for st in abstack.chunk(chunks=4, dim=0):
        print(itr, st.shape)
        itr += 1
