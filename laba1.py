#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

L = 3

parser = argparse.ArgumentParser()

parser.add_argument('-x', type=float, help="X-coordinate")
parser.add_argument('-y', type=float, help="Y-coordinate")
parser.add_argument('-z', type=float, help="Z-coordinate")
parser.add_argument('-obj_to_obj_method', type=int)
parser.add_argument('-obj_to_class_method', type=int)
args = parser.parse_args(sys.argv[1:])
MY_X = args.x
MY_Y = args.y
MY_Z = args.z

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
    x = class_points[:, 0]
    y = class_points[:, 1]
    z = class_points[:, 2]
    return [sum(x) / len(class_points), sum(y) / len(class_points), sum(z) / len(class_points)]


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


def input_choose_function_dict():
    """
    Возвращает метод вычисления расстрояния между двумя объектами
    :return: function
    """

    def get_input():
        print("Выберите метод вычисления расстрояния между двумя объектами в двумерном пространстве:")
        print("1. Евклидово расстояния")
        print("2. Расстрояние Минковского")
        obj_to_obj_method = int(input())
        if obj_to_obj_method not in [1, 2]:
            return get_input()
        else:
            return str(obj_to_obj_method)

    method = {'1': euclid_dist,
              '2': minkovsky_dist}

    return method.get(get_input())


def input_choose_method_dict():
    """
    Возвращает метод вычисления расстрояния между объектом и классом
    :return: function
    """

    def get_input():
        print("Выберите метод вычисления расстрояния между объектом и классом:")
        print("1. Расстояние до центроида класса")
        print("2. Найменше з значень відстані до усіх еталонів класу(«найближчий сусід»)")
        obj_to_obj_method = int(input())
        if obj_to_obj_method not in [1, 2]:
            return get_input()
        else:
            return str(obj_to_obj_method)

    method = {'1': get_distance_to_centroid,
              '2': get_distance_to_nearest_neighbor}

    return method.get(get_input())


def standardize(class_list):
    """
    Выполняет стандартизацию признаков
    :param class_list: список классов
    :return:
    """
    x_max_value = max(np.concatenate(class_list)[:, 0])
    x_min_value = min(np.concatenate(class_list)[:, 0])
    y_max_value = max(np.concatenate(class_list)[:, 1])
    y_min_value = min(np.concatenate(class_list)[:, 1])
    z_max_value = max(np.concatenate(class_list)[:, 2])
    z_min_value = min(np.concatenate(class_list)[:, 2])

    for class_ in class_list:
        for point in class_:
            point[0] = (point[0] - x_min_value)/(x_max_value - x_min_value)
            point[1] = (point[1] - y_min_value) / (y_max_value - y_min_value)
            point[2] = (point[2] - z_min_value)/(z_max_value - z_min_value)

    return class_list


class1, class2, class3, class4 = standardize(class_list=np.array([class1, class2, class3, class4]))

fig1 = plt.figure()
ax = fig1.add_subplot(projection='3d')
ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c=class_colors.get('1'), label='class 1')
ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c=class_colors.get('2'), label='class 2', marker='^')
ax.scatter(class3[:, 0], class3[:, 1], class3[:, 2], c=class_colors.get('3'), label='class 3', marker='s')
ax.scatter(class4[:, 0], class4[:, 1], class4[:, 2], c=class_colors.get('4'), label='class 4', marker='d')

ax.scatter(MY_X, MY_Y, MY_Z, c='k', label='new point', marker='p')
ax.legend()

indices = np.arange(0, 1, 0.05)

fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')
class_list = {'1': class1, '2': class2, '3': class3, '4': class4}
i = 0
for x in indices:
    for y in indices:
        for z in indices:
            nearest_class = find_nearest_class(class_list, (x, y, z), get_distance_to_centroid, euclid_dist).__iter__().__next__()
            ax.scatter(x, y, z, c=class_colors[nearest_class], alpha=0.15)
            i += 1
            if (i % 100) == 0:
                print(str(i / ((11 / 0.1) ** 3)), end='\r')
                # print(i)

plt.show()
