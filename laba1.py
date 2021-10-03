#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

L = 5

parser = argparse.ArgumentParser()

parser.add_argument('--first_point', type=float, help="first point -coordinate", default=None, nargs='+')
parser.add_argument('--second_point', type=float, help="second point -coordinate", default=None, nargs='+')
parser.add_argument('--third_point', type=float, help="third point -coordinate", default=None, nargs='+')
parser.add_argument('--dist', type=str, choices=['1', '2'],
                    help="Choose dist (1. Евклидово расстояния, 2. Расстрояние Минковского", default=None, )
parser.add_argument('--distance_to', type=str, choices=['1', '2'],
                    help="Choose distance_to (1. Расстояние до центроида класса, 2. Найменше з значень відстані "
                         "до усіх еталонів класу(«найближчий сусід»))",
                    default=None, )

namespace = parser.parse_args(sys.argv[1:])

if namespace.first_point and len(namespace.first_point) != 3:
    raise ValueError('Некорректные координаты первой точки')
if namespace.second_point and len(namespace.second_point) != 3:
    raise ValueError('Некорректные координаты второй точки')
if namespace.third_point and len(namespace.third_point) != 3:
    raise ValueError('Некорректные координаты третьей точки')

class1 = np.array([[2.6, 3.3, 3.7],
                   [3.2, 3.8, 2.7],
                   [2.6, 4.1, 3.6],
                   [2.1, 3.0, 2.6],
                   [3.0, 4.1, 3.2],
                   [2.5, 3.1, 2.6],
                   [3.6, 3.9, 2.4],
                   [2.8, 4.5, 3.5],
                   [3.8, 3.3, 4.0],
                   [3.9, 3.2, 2.9],
                   [2.1, 4.8, 2.2],
                   [3.8, 5.0, 2.4],
                   [4.0, 3.9, 3.7],
                   [2.1, 4.3, 2.1],
                   [2.5, 5.0, 2.5],
                   [2.2, 4.0, 2.9],
                   [2.1, 4.6, 2.2],
                   [3.5, 3.9, 2.4],
                   [2.2, 4.8, 2.5],
                   [3.8, 3.3, 3.6]])

class2 = np.array([[5.6, 9.5, 3.2],
                   [5.9, 9.1, 3.5],
                   [6.6, 9.3, 3.0],
                   [6.8, 8.4, 4.8],
                   [6.1, 8.5, 3.8],
                   [6.5, 9.9, 4.9],
                   [5.9, 9.2, 4.2],
                   [5.3, 8.8, 4.6],
                   [6.8, 9.9, 4.7],
                   [5.9, 9.1, 3.4],
                   [5.6, 8.1, 3.8],
                   [5.8, 8.5, 3.4],
                   [6.9, 9.4, 4.7],
                   [5.7, 9.5, 4.5],
                   [6.6, 9.3, 4.2],
                   [5.0, 8.5, 4.1],
                   [5.0, 9.5, 4.0],
                   [6.0, 8.8, 3.6],
                   [5.8, 9.5, 4.7],
                   [6.4, 8.8, 3.6]])

class3 = np.array([[10.5, 10.4, 10.8],
                   [10.6, 10.8, 10.3],
                   [11.0, 11.9, 9.4],
                   [10.0, 11.7, 9.6],
                   [10.3, 10.2, 9.0],
                   [10.3, 10.7, 9.7],
                   [10.6, 11.5, 9.1],
                   [9.8, 11.3, 10.9],
                   [10.0, 11.6, 9.2],
                   [9.7, 11.7, 10.7],
                   [9.3, 10.8, 9.5],
                   [10.7, 11.3, 10.5],
                   [10.7, 11.3, 10.5],
                   [10.2, 11.4, 10.2],
                   [9.3, 11.7, 9.3],
                   [9.9, 10.1, 9.8],
                   [9.2, 10.1, 10.1],
                   [9.6, 11.2, 10.9],
                   [10.3, 11.5, 10.4],
                   [9.1, 11.8, 10.3]])

class4 = np.array([[3.7, 3.7, 9.3],
                   [4.4, 2.0, 10.7],
                   [4.2, 4.0, 10.2],
                   [3.2, 3.5, 10.7],
                   [4.0, 3.6, 9.4],
                   [4.8, 3.7, 9.8],
                   [3.1, 2.5, 10.3],
                   [4.4, 2.4, 9.9],
                   [4.0, 3.4, 9.4],
                   [4.4, 2.7, 9.1],
                   [4.0, 2.6, 9.0],
                   [4.0, 2.8, 9.0],
                   [4.5, 2.9, 9.6],
                   [3.3, 2.5, 10.9],
                   [3.0, 2.9, 9.5],
                   [3.8, 3.9, 10.4],
                   [4.0, 3.5, 10.7],
                   [4.2, 2.8, 9.8],
                   [4.7, 2.3, 10.2],
                   [3.4, 3.5, 9.9]])

class_colors = {
    '1': 'r',
    '2': 'b',
    '3': 'g',
    '4': 'y'
}

class_markers = {
    '1': 'o',
    '2': '^',
    '3': 's',
    '4': 'd'
}


def check_value(*args):
    for val in args:
        if val < -5 or val > 15:
            return False
    return True


def minkovsky_dist(o1, o2, k=L):
    """
    Функция для нахождения расстояния Минковского между двумя точками
    :param o1: координаты первой точки
    :param o2: координаты второй точки
    :param k: коэффициент
    :return: расстояние между точками о1 и о2
    """
    if not isinstance(k, int) or k < 2:
        raise ValueError("k should be integer, k>2")
    dx = abs(o1[0] - o2[0]) ** k
    dy = abs(o1[1] - o2[1]) ** k
    dz = abs(o1[2] - o2[2]) ** k
    return (dx + dy + dz) ** (1 / k)


def euclid_dist(o1, o2):
    """
    Функция для нахождения расстояния Эвклида между двумя точками
    :param o1: координаты первой точки
    :param o2: координаты второй точки
    :return: расстояние между точками о1 и о2
    """
    dx = (o1[0] - o2[0]) ** 2
    dy = (o1[1] - o2[1]) ** 2
    dz = (o1[2] - o2[2]) ** 2
    d_euclid = np.sqrt(dx + dy + dz)
    return d_euclid


def get_centroid(class_points):
    """
    Рассчитывает центроид класса
    :param class_points: список координат точек класса
    :return: координаты центроида класса
    """
    _x = class_points[:, 0]
    _y = class_points[:, 1]
    _z = class_points[:, 2]
    return [sum(_x) / len(class_points), sum(_y) / len(class_points), sum(_z) / len(class_points)]


def get_distance_to_centroid(o1, class_points, distance_fun, *args):
    """
    Функция рассчитывает расстояние от точки o1 до центроида класса
    :param o1: координаты точки
    :param class_points: список координат точек класса
    :param distance_fun: функция для расчета расстояния
    :return: расстояние
    """
    centroid = get_centroid(class_points)
    distance = distance_fun(o1, centroid, *args)
    return distance


def get_distance_to_nearest_neighbor(o1, class_points, distance_fun, *args):
    """
    Функция рассчитывает расстояние от точки o1 до ближайшей точки другого класса
    :param o1: координаты точки
    :param class_points: список координат точек класса
    :param distance_fun: функция для расчета расстояния
    :return: расстояние
    """
    distance = float("inf")
    for point in class_points:
        if distance_fun(o1, point, *args) < distance:
            distance = distance_fun(o1, point, *args)
    return distance


def find_nearest_class(classes_list, o1, func_dist_to_class, func_dist_point_to_point) -> dict:
    """
    Определяет класс к которому относиться объект, возвращает номер класса
    :param classes_list: список классов
    :param o1: координаты точки
    :param func_dist_to_class: метод вычисления расстояния между точкой и классом
    :param func_dist_point_to_point: метод вычисления расстояния между точками
    :return: номер класса
    """
    list_of_distances = sorted(
        {
            class_name: func_dist_to_class(o1, class_points, func_dist_point_to_point)
            for class_name, class_points in classes_list.items()
        }.items(),
        key=lambda value: value[1])
    return dict(list_of_distances)


def input_choose_method_obj_to_obj(dist=None):
    """
    Возвращает метод вычисления расстрояния между двумя объектами
    :return: function
    """

    def get_input():
        print("1. Евклидово расстояния")
        print("2. Расстрояние Минковского")
        obj_to_obj_method = int(input())
        return get_input() if obj_to_obj_method not in [1, 2] else str(obj_to_obj_method)

    print("Выберите метод вычисления расстрояния между двумя объектами в двумерном пространстве:")
    method = {'1': euclid_dist,
              '2': minkovsky_dist}
    _dist = dist if dist else get_input()
    print("Евклидово расстояния") if _dist == '1' else print("Расстрояние Минковского")
    return method.get(_dist)


def input_choose_method_obj_to_class(distance_to=None):
    """
    Возвращает метод вычисления расстрояния между объектом и классом
    :return: function
    """

    def get_input():
        print("1. Расстояние до центроида класса")
        print("2. Найменшее из значений расстояния ко всем еталонам класса («ближайший сосед»)")
        obj_to_class_method = int(input())
        return get_input() if obj_to_class_method not in [1, 2] else str(obj_to_class_method)

    print("Выберите метод вычисления расстрояния между объектом и классом:")
    method = {'1': get_distance_to_centroid,
              '2': get_distance_to_nearest_neighbor}

    _distance_to = distance_to if distance_to else get_input()
    print("Расстояние до центроида класса") if _distance_to == '1' else print("Наименьшее из значений расстояния ко всем "
                                                                              "эталонам класса («ближайший сосед»)")
    return method.get(_distance_to)


def input_new_point(message=None):
    """
    Функция для ввода координат точки
    :return: function
    """
    print("Введите координаты точки " + message + ":\t")
    try:
        _x = float(input("x: "))
        _y = float(input("y: "))
        _z = float(input("z: "))
        if check_value(_x, _y, _z):
            return _x, _y, _z
        else:
            print("Value error")
            return input_new_point(message)
    except ValueError:
        print("Value error")
        input_new_point(message)


def standardize(class_list, new_points=None):
    """
    Выполняет стандартизацию признаков
    :param new_points: новые точки, введенные пользователем
    :param class_list: список классов
    :return:
    """
    if new_points is not None:
        x_max_value = np.max(np.concatenate([np.concatenate(class_list)[:, 0], new_points[:, 0]]))
        x_min_value = np.min(np.concatenate([np.concatenate(class_list)[:, 0], new_points[:, 0]]))
        y_max_value = np.max(np.concatenate([np.concatenate(class_list)[:, 1], new_points[:, 1]]))
        y_min_value = np.min(np.concatenate([np.concatenate(class_list)[:, 1], new_points[:, 1]]))
        z_max_value = np.max(np.concatenate([np.concatenate(class_list)[:, 2], new_points[:, 2]]))
        z_min_value = np.min(np.concatenate([np.concatenate(class_list)[:, 2], new_points[:, 2]]))
        for point in new_points:
            point[0] = (point[0] - x_min_value) / (x_max_value - x_min_value)
            point[1] = (point[1] - y_min_value) / (y_max_value - y_min_value)
            point[2] = (point[2] - z_min_value) / (z_max_value - z_min_value)
    else:
        x_max_value = np.max(np.concatenate(class_list)[:, 0])
        x_min_value = np.min(np.concatenate(class_list)[:, 0])
        y_max_value = np.max(np.concatenate(class_list)[:, 1])
        y_min_value = np.min(np.concatenate(class_list)[:, 1])
        z_max_value = np.max(np.concatenate(class_list)[:, 2])
        z_min_value = np.min(np.concatenate(class_list)[:, 2])

    for class_ in class_list:
        for point in class_:
            point[0] = (point[0] - x_min_value) / (x_max_value - x_min_value)
            point[1] = (point[1] - y_min_value) / (y_max_value - y_min_value)
            point[2] = (point[2] - z_min_value) / (z_max_value - z_min_value)

    if new_points is None:
        return class_list
    else:
        return class_list, new_points


def display_classes(class_list, _ax):
    class1 = class_list[0]
    class2 = class_list[1]
    class3 = class_list[2]
    class4 = class_list[3]
    _ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c=class_colors.get('1'), marker=class_markers['1'],
               label='class 1')
    _ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c=class_colors.get('2'), marker=class_markers['2'],
               label='class 2')
    _ax.scatter(class3[:, 0], class3[:, 1], class3[:, 2], c=class_colors.get('3'), marker=class_markers['3'],
               label='class 3')
    _ax.scatter(class4[:, 0], class4[:, 1], class4[:, 2], c=class_colors.get('4'), marker=class_markers['4'],
               label='class 4')


if __name__ == '__main__':
    obj_to_obj_method = euclid_dist
    obj_to_class_method = get_distance_to_centroid

    x_1, y_1, z_1 = namespace.first_point if check_value(*namespace.first_point) else input_new_point("№1")
    print(f"Координаты первой точки: ({x_1}, {y_1}, {z_1})")
    x_2, y_2, z_2 = namespace.second_point if check_value(*namespace.second_point) else input_new_point("№2")
    print(f"Координаты второй точки: ({x_2}, {y_2}, {z_2})")
    x_3, y_3, z_3 = namespace.third_point if check_value(*namespace.third_point) else input_new_point("№3")
    print(f"Координаты третьей точки: ({x_3}, {y_3}, {z_3})")

    my_obj_to_obj_method = input_choose_method_obj_to_obj(namespace.dist)
    my_obj_to_class_method = input_choose_method_obj_to_class(namespace.distance_to)

    # отображение классов на графике без стандартизации признаков
    fig_classes_no_standardization = plt.figure()
    ax = fig_classes_no_standardization.add_subplot(projection='3d')
    ax.set_title('Отображение 4 классов в начальном пространстве', pad=25)
    display_classes([class1, class2, class3, class4], ax)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # отображение новых точек на графике
    fig_classes_no_standardization_new_points = plt.figure()
    ax = fig_classes_no_standardization_new_points.add_subplot(projection='3d')
    ax.set_title('Отображение неизвестных объектов в начальном пространстве', pad=25)
    display_classes([class1, class2, class3, class4], ax)
    ax.scatter(x_1, y_1, z_1, c='k', label='x 1', marker='p')
    ax.scatter(x_2, y_2, z_2, c='k', label='x 2', marker='p')
    ax.scatter(x_3, y_3, z_3, c='k', label='x 3', marker='p')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #
    # стандартизация признаков
    (class1, class2, class3, class4), ((x_1, y_1, z_1), (x_2, y_2, z_2), (x_3, y_3, z_3)) = standardize(class_list=np.array([class1, class2, class3, class4]),
                                                 new_points=np.array([[x_1, y_1, z_1], [x_2, y_2, z_2], [x_3, y_3, z_3]]))

    # отображение классов на графике после стандартизации признаков
    fig_classes_with_standardization = plt.figure()
    ax = fig_classes_with_standardization.add_subplot(projection='3d')
    ax.set_title('Отображение 4 классов после стандаризации признаков', pad=25)
    display_classes([class1, class2, class3, class4], ax)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # отображение новых точек на графике после стандартизации признаков
    fig_classes_with_standardization_new_points = plt.figure()
    ax = fig_classes_with_standardization_new_points.add_subplot(projection='3d')
    ax.set_title('Отображение неизвестных объектов после стандартизации признаков', pad=25)
    display_classes([class1, class2, class3, class4], ax)
    ax.scatter(x_1, y_1, z_1, c='k', label='new point 1', marker='p')
    ax.scatter(x_2, y_2, z_2, c='k', label='new point 2', marker='p')
    ax.scatter(x_3, y_3, z_3, c='k', label='new point 3', marker='p')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    # определение принадлежности новых точек к классам
    fig_new_points_classes = plt.figure()
    ax = fig_new_points_classes.add_subplot(projection='3d')
    display_classes([class1, class2, class3, class4], ax)
    ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c=class_colors.get('1'), label='class 1')
    ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c=class_colors.get('2'), label='class 2', marker='^')
    ax.scatter(class3[:, 0], class3[:, 1], class3[:, 2], c=class_colors.get('3'), label='class 3', marker='s')
    ax.scatter(class4[:, 0], class4[:, 1], class4[:, 2], c=class_colors.get('4'), label='class 4', marker='d')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    my_class_list = {'1': class1, '2': class2, '3': class3, '4': class4}
    point_1_class = find_nearest_class(my_class_list, (x_1, y_1, z_1), my_obj_to_class_method,
                                       my_obj_to_obj_method).__iter__().__next__()
    ax.scatter(x_1, y_1, z_1, c=class_colors[point_1_class], marker='p')

    point_2_class = find_nearest_class(my_class_list, (x_2, y_2, z_2), my_obj_to_class_method,
                                       my_obj_to_obj_method).__iter__().__next__()
    ax.scatter(x_2, y_2, z_2, c=class_colors[point_2_class], marker='p')

    point_3_class = find_nearest_class(my_class_list, (x_3, y_3, z_3), my_obj_to_class_method,
                                       my_obj_to_obj_method).__iter__().__next__()
    ax.scatter(x_3, y_3, z_3, c=class_colors[point_3_class], marker='p')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    indices = np.arange(0, 1, 0.05)

    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')
    i = 0
    for x in indices:
        for y in indices:
            for z in indices:
                nearest_class = find_nearest_class(my_class_list, (x, y, z), obj_to_class_method,
                                                   obj_to_obj_method).__iter__().__next__()
                ax.scatter(x, y, z, c=class_colors[nearest_class], alpha=0.15)
                i += 1
                if (i % 100) == 0:
                    print(str(i / ((11 / 0.1) ** 3)), end='\r')
                    # print(i)
    ax.set_title('Классификация точек пространства\n'
                 'Эвклидово расстояние\n'
                 'Расстояние до "ближайшего соседа"', pad=25)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
