import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

X_ARRAY = np.arange(-2, 3, 0.1)
Y_ARRAY = np.arange(-0.5, 2, 0.1)

class_colors = {
    '1': 'r',
    '2': 'b',
    '3': 'g',
}

parser = argparse.ArgumentParser()
parser.add_argument('--d12_A', type=float, default=1)
parser.add_argument('--d12_B', type=float, default=-1)
parser.add_argument('--d12_C', type=float, default=-0.1)
parser.add_argument('--d13_D', type=float, default=2)
parser.add_argument('--d13_E', type=float, default=1)
parser.add_argument('--d13_F', type=float, default=-1.2)
parser.add_argument('--d23_G', type=float, default=1)
parser.add_argument('--d23_H', type=float, default=6)
parser.add_argument('--d23_K', type=float, default=-5)
parser.add_argument('--d1_M', type=float, default=-1)
parser.add_argument('--d1_N', type=float, default=1)
parser.add_argument('--d1_P', type=float, default=-0.3)
parser.add_argument('--d2_Q', type=float, default=1)
parser.add_argument('--d2_R', type=float, default=1)
parser.add_argument('--d2_S', type=float, default=-1.2)
parser.add_argument('--d3_V', type=float, default=-1)
parser.add_argument('--d3_W', type=float, default=-3)
parser.add_argument('--d3_Z', type=float, default=1.7)

namespace = parser.parse_args(sys.argv[1:])

D12_A = namespace.d12_A
D12_B = namespace.d12_B
D12_C = namespace.d12_C
D13_D = namespace.d13_D
D13_E = namespace.d13_E
D13_F = namespace.d13_F
D23_G = namespace.d23_G
D23_H = namespace.d23_H
D23_K = namespace.d23_K
D1_M = namespace.d1_M
D1_N = namespace.d1_N
D1_P = namespace.d1_P
D2_Q = namespace.d2_Q
D2_R = namespace.d2_R
D2_S = namespace.d2_S
D3_V = namespace.d3_V
D3_W = namespace.d3_W
D3_Z = namespace.d3_Z

print(namespace)


def find_cos(x1, y1, x2, y2):
    return (x1 * x2 + y1 * y2) / ((np.sqrt(x1 ** 2 + y1 ** 2)) * np.sqrt(x2 ** 2 + y2 ** 2))


def define_class(x, y):
    """
    Определяет номер класса, к которому принадлежит точка o1
    :param o1: координаты точки
    :return: номер класса
    """

    d12 = lambda x, y: (D12_A * x + D12_B * y + D12_C) > 0
    d13 = lambda x, y: (D13_D * x + D13_E * y + D13_F) > 0
    d23 = lambda x, y: (D23_G * x + D23_H * y + D23_K) > 0

    if d12(x, y) and d13(x, y):
        return '1'
    elif not d12(x, y) and d23(x, y):
        return '2'
    elif not d23(x, y) and not d13(x, y):
        return '3'
    else:
        return 'undefined'


# точки для прорисовки прямых решающих функций
fig, ax = plt.subplots()
ax.set_xlim([min(X_ARRAY), max(X_ARRAY)])
ax.set_ylim([min(Y_ARRAY), max(Y_ARRAY)])
x1_d12 = X_ARRAY[0]
y1_d12 = (-D12_A * x1_d12 - D12_C) / D12_B
x2_d12 = X_ARRAY[-1]
y2_d12 = (-D12_A * x2_d12 - D12_C) / D12_B
ax.plot([x1_d12, x2_d12], [y1_d12, y2_d12], c='k', linestyle=':', linewidth=2.5)
print(find_cos(x1_d12, y1_d12, x2_d12, y2_d12))

x1_d13 = X_ARRAY[0]
y1_d13 = (-D13_D * x1_d13 - D13_F) / D13_E
x2_d13 = X_ARRAY[-1]
y2_d13 = (-D13_D * x2_d13 - D13_F) / D13_E
ax.plot([x1_d13, x2_d13], [y1_d13, y2_d13], c='k', linestyle='--', linewidth=2.5)

x1_d23 = X_ARRAY[0]
y1_d23 = (-D23_G * x1_d23 - D23_K) / D23_H
x2_d23 = X_ARRAY[-1]
y2_d23 = (-D23_G * x2_d23 - D23_K) / D23_H
plt.plot([x1_d23, x2_d23], [y1_d23, y2_d23], c='k', linestyle='-.', linewidth=2.5)

ax.text(1.5, 1.6, 'd12', rotation=y1_d12 / x1_d12 * 45, fontsize='x-large')

X_ARRAY = np.arange(-2, 3, 0.1)
Y_ARRAY = np.arange(-0.5, 2, 0.1)

for x in X_ARRAY:
    for y in Y_ARRAY:
        class_ = define_class(x, y)
        if class_ != 'undefined':
            plt.scatter(x, y, c=class_colors[class_], alpha=1, s=8)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
plt.show()
