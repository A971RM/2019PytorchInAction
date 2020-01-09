"""
f(x,y) = (x**2+y-11)**2+(x+y**2-7)**2
"""
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def test_himmelblau():
    x = np.arange(-6, 6, 0.1, dtype=np.float)
    y = np.arange(-6, 6, 0.1, dtype=np.float)
    print("x, y shape: ", x.shape, y.shape)
    X, Y = np.meshgrid(x, y)
    print("X, Y maps:", X.shape, Y.shape)
    Z = himmelblau(X, Y)
    print("Z shape: ", Z.shape)

    fig = plt.figure(num='himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()

def test_optim():
    x = torch.tensor([0., 0.], requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=1e-3)
    for step in range(1, 20001):
        pred = himmelblau(x[0], x[1])

        optimizer.zero_grad()
        pred.backward()
        optimizer.step()

        if step % 200 == 0:
            print('step {}: x={}, f(x)={}'.format(step, x.tolist(), pred.item()))

if __name__ == "__main__":
    # test_himmelblau()
    test_optim()