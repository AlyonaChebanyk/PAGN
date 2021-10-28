import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from laba1 import class1, class2, class3, class4
from laba1 import display_classes, get_centroid, standardize
from laba1 import class_colors as cl
from labellines import labelLine, labelLines


def get_pseudo_inverse_matrix(class_1, class_2):
    class_1 = np.column_stack((class_1, [1] * len(class_1)))
    class_2 = np.column_stack((class_2, [1] * len(class_2))) * -1
    V = np.row_stack((class_1, class_2))
    V_x = lambda matrix: np.dot(
        np.linalg.matrix_power(
            np.dot(matrix.transpose(), matrix), -1),
        matrix.transpose()
    )

    y_ = np.array([1] * len(V_x(V)[0]))

    def cheek_result(V, y_):
        w_ = lambda y: np.dot(V_x(V), y)
        V_w = np.dot(V, w_(y_))
        if V_w.all() > 0:
            return w_(y_)

        # k = 0
        h_k = 0.5
        return cheek_result(V, y_ + h_k * (V_w - y_))

    return cheek_result(V, y_)


def draw_area(class_1, class_2, ax, bords: np.ndarray = np.linspace(0, 1, 10)):
    pseudo_inverse_matrix = get_pseudo_inverse_matrix(class_1, class_2)
    X, Y = np.meshgrid(bords, bords)
    Z = lambda x, y, pim: (pim[0] * x + pim[1] * y + pim[3]) / (-pim[2])
    ax.plot_surface(X, Y, Z(X, Y, pseudo_inverse_matrix), alpha=0.2)


def define_class_3d(X, Y, Z):
    d_x = lambda x, y, z, pim: (pim[0] * x + pim[1] * y + pim[2] * z + pim[3])
    d12 = d_x(X, Y, Z, get_pseudo_inverse_matrix(class1, class2)) > 0
    d13 = d_x(X, Y, Z, get_pseudo_inverse_matrix(class1, class3)) > 0
    d14 = d_x(X, Y, Z, get_pseudo_inverse_matrix(class1, class4)) > 0
    d23 = d_x(X, Y, Z, get_pseudo_inverse_matrix(class2, class3)) > 0
    d24 = d_x(X, Y, Z, get_pseudo_inverse_matrix(class2, class4)) > 0
    d34 = d_x(X, Y, Z, get_pseudo_inverse_matrix(class3, class4)) > 0

    if d12 and d13 and d14:
        return '1'
    elif not d12 and d23 and d24:
        return '2'
    elif not d13 and not d23 and d34:
        return '3'
    elif not d14 and not d24 and not d34:
        return '4'
    else:
        return 'undefined'


def check_value(*args):
    """
    Проверка значений координат (val < 0 or val > 1)
    :param args: координаты
    :return:
    """
    for val in args:
        if val < 0 or val > 1:
            return False
    return True


def input_new_point():
    """
    Функция для ввода координат точки
    :return: function
    """
    print("Введите координаты точки " + ":\t")
    try:
        _x = float(input("x: "))
        _y = float(input("y: "))
        _z = float(input("z: "))
        if check_value(_x) and check_value(_y) and check_value(_z):
            return _x, _y, _z
        else:
            print("Value error")
            return input_new_point()
    except ValueError:
        print("Value error")
        input_new_point()


fig_classes_no_standardization = plt.figure()
ax = fig_classes_no_standardization.add_subplot(projection='3d')
(class1, class2, class3, class4) = standardize(class_list=np.array([class1, class2, class3, class4]))
display_classes([class1, class2, class3, class4], ax)

draw_area(class1, class2, ax)
draw_area(class1, class3, ax)
draw_area(class1, class4, ax)
draw_area(class2, class3, ax)
draw_area(class2, class4, ax)
draw_area(class3, class4, ax)

ax.axes.set_xlim3d(0, 1)
ax.axes.set_ylim3d(0, 1)
ax.axes.set_zlim3d(0, 1)

bord = np.linspace(0, 1, 10)

fig_classes_no_standardization = plt.figure()
ax = fig_classes_no_standardization.add_subplot(projection='3d')
for x in bord:
    for y in bord:
        for z in bord:
            class_ = define_class_3d(x, y, z)
            if class_ != 'undefined':
                ax.scatter(x, y, z, c=cl[class_], alpha=0.5)

ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x_input, y_input, z_input = input_new_point()
defined_class = define_class_3d(x_input, y_input, z_input)

if defined_class == 'undefined':
    print('Точку нельзя отнести ни к одному классу')
else:
    print(f'Точка относится к классу {defined_class}')

    fig_classes_no_standardization = plt.figure()
    ax = fig_classes_no_standardization.add_subplot(projection='3d')
    draw_area(class1, class2, ax)
    draw_area(class1, class3, ax)
    draw_area(class1, class4, ax)
    draw_area(class2, class3, ax)
    draw_area(class2, class4, ax)
    draw_area(class3, class4, ax)
    ax.scatter(x_input, y_input, c=cl[defined_class], marker='p', s=70)
    ax.axes.set_xlim3d(0, 1)
    ax.axes.set_ylim3d(0, 1)
    ax.axes.set_zlim3d(0, 1)

plt.show()
