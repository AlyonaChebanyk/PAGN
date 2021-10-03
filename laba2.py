import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from labellines import labelLine, labelLines

X_ARRAY = np.arange(-2, 3, 0.15)
Y_ARRAY = np.arange(-1, 2.5, 0.15)

class_colors = {
    '1': 'r',
    '2': 'b',
    '3': 'g',
    'undefined': 'k'
}

class_markers = {
    '1': 'o',
    '2': '^',
    '3': 's',
    'undefined': 'o'
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


def define_class_2(x, y):
    """
    Определяет номер класса, к которому принадлежит точка o1
    :param o1: координаты точки
    :return: номер класса
    """

    d1 = lambda x, y: (D1_M * x + D1_N * y + D1_P) > 0
    d2 = lambda x, y: (D2_Q * x + D2_R * y + D2_S) > 0
    d3 = lambda x, y: (D3_V * x + D3_W * y + D3_Z) > 0

    if d1(x, y) and not d2(x, y) and not d3(x, y):
        return '1'
    elif not d1(x, y) and d2(x, y) and not d3(x, y):
        return '2'
    elif not d1(x, y) and not d2(x, y) and d3(x, y):
        return '3'
    else:
        return 'undefined'


def plot_decisive_function(coef1, coef2, coef3, x_lim, _ax, c='k', linestyle='-', linewidth=1, label=None):
    """
    Функция рисует решающую функцию
    :param coef1: коэффициент перед х
    :param coef2: коэффициент перед y
    :param coef3: сводобный коэффициент
    :param x_lim: границы значений x
    :param c: цвет линии
    :param linestyle: стиль линии
    :param linewidth: ширина линии
    :return: координаты 2-ух точек линии
    """
    x1 = x_lim[0]
    y1 = (-coef1 * x1 - coef3) / coef2
    x2 = x_lim[-1]
    y2 = (-coef1 * x2 - coef3) / coef2
    _ax.plot([x1, x2], [y1, y2], c=c, linestyle=linestyle, linewidth=linewidth, label=label)


def input_new_point(message=""):
    """
    Функция для ввода координат точки
    :return: function
    """
    print("Введите координаты точки " + message + ":\t")
    try:
        _x = float(input("x: "))
        _y = float(input("y: "))
        if check_value_x(_x) and check_value_y(_y):
            return _x, _y
        else:
            print("Value error")
            return input_new_point(message)
    except ValueError:
        print("Value error")
        input_new_point(message)


def check_value_x(*args):
    """
    Проверка значений координат (val < -5 or val > 15)
    :param args: координаты
    :return:
    """
    for val in args:
        if val < -1.7 or val > 2.7:
            return False
    return True


def check_value_y(*args):
    """
    Проверка значений координат (val < -5 or val > 15)
    :param args: координаты
    :return:
    """
    for val in args:
        if val < -0.4 or val > 1.6:
            return False
    return True


# точки для прорисовки прямых решающих функций
def plot_classes_1():
    fig, ax = plt.subplots()

    ax.set_xlim([min(X_ARRAY), max(X_ARRAY)])
    ax.set_ylim([min(Y_ARRAY), max(Y_ARRAY)])

    plot_decisive_function(D12_A, D12_B, D12_C, X_ARRAY, ax)
    plot_decisive_function(D13_D, D13_E, D13_F, X_ARRAY, ax)
    plot_decisive_function(D23_G, D23_H, D23_K, X_ARRAY, ax)
    lines = plt.gca().get_lines()
    l1 = lines[-3]
    l2 = lines[-2]
    l3 = lines[-1]
    labelLine(l1, 1.6, label='d12', align=False, fontsize=14)
    labelLine(l2, 0.7, label='d13', align=False, fontsize=14)
    labelLine(l3, -1, label='d23', align=False, fontsize=14)

    for x in X_ARRAY:
        for y in Y_ARRAY:
            class_ = define_class(x, y)
            if class_ != 'undefined':
                plt.scatter(x, y, c=class_colors[class_], marker=class_markers[class_], alpha=0.8, s=8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    return ax


def plot_classes_2():
    fig, ax = plt.subplots()

    ax.set_xlim([min(X_ARRAY), max(X_ARRAY)])
    ax.set_ylim([min(Y_ARRAY), max(Y_ARRAY)])

    plot_decisive_function(D1_M, D1_N, D1_P, X_ARRAY, ax)
    plot_decisive_function(D2_Q, D2_R, D2_S, X_ARRAY, ax)
    plot_decisive_function(D3_V, D3_W, D3_Z, X_ARRAY, ax)
    lines = plt.gca().get_lines()
    l1 = lines[-3]
    l2 = lines[-2]
    l3 = lines[-1]
    labelLine(l1, 1, label='d1', align=False, fontsize=14)
    labelLine(l2, 0, label='d2', align=False, fontsize=14)
    labelLine(l3, 2, label='d3', align=False, fontsize=14)

    for x in X_ARRAY:
        for y in Y_ARRAY:
            class_ = define_class_2(x, y)
            if class_ != 'undefined':
                plt.scatter(x, y, c=class_colors[class_], marker=class_markers[class_], alpha=1, s=8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    return ax


x_input, y_input = input_new_point()
print("Выберете способ классификации с помощью решающих функций:")
while True:
    try:
        print("1 - Решающие функции отделяют класс от каждого другого класса")
        print("2 - Решающая функция отделяет класс от всех остальных классов")
        class_def_way = int(input("Способ: "))
        if class_def_way not in [1, 2]:
            print("Недопустимое значение")
        else:
            break
    except ValueError:
        print("Недопустимое значение")

if class_def_way == 1:
    ax = plot_classes_1()
    defined_class = define_class(x_input, y_input)
else:
    ax = plot_classes_2()
    defined_class = define_class_2(x_input, y_input)

if defined_class == 'undefined':
    print('Точку нельзя отнести ни к одному классу')
else:
    print(f'Точка относится к классу {defined_class}')

ax.scatter(x_input, y_input, c='k', marker='p', s=70)


# ax = plot_classes_1()
# ax.scatter(-100, -100, c=class_colors['1'], marker=class_markers['1'], label='class 1')
# ax.scatter(-100, -100, c=class_colors['2'], marker=class_markers['2'], label='class 2')
# ax.scatter(-100, -100, c=class_colors['3'], marker=class_markers['3'], label='class 3')
# ax.legend(loc='upper right')
# ax = plot_classes_2()
ax.scatter(-100, -100, c=class_colors['1'], marker=class_markers['1'], label='class 1')
ax.scatter(-100, -100, c=class_colors['2'], marker=class_markers['2'], label='class 2')
ax.scatter(-100, -100, c=class_colors['3'], marker=class_markers['3'], label='class 3')
ax.legend(loc='upper right')
plt.show()