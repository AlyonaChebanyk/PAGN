#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

L = 3

parser = argparse.ArgumentParser()

parser.add_argument('-x', type=int)
parser.add_argument('-y', type=int)
parser.add_argument('-z', type=int)
args = parser.parse_args(sys.argv[1:])
print(args)
MY_X = args.x
MY_Y = args.y
MY_Z = args.z
print(MY_X, MY_Y, MY_Z)

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


class_colors = {
    1: 'r',
    2: 'b',
    3: 'g',
    4: 'm'
}

# print("Расстояние Минковского:")
# print(minkovsky_dist(class1[0], class2[0], 4))
# print("Эвклидово расстояние:")
# print(euclid_dist(class1[0], class2[0]))
#
# print("Расстояние до центроида:")
# print(get_distance_to_centroid(class1[0], class2, euclid_dist))
# print("Расстояние до ближайшего соседа:")
# print(get_distance_to_nearest_neighbor(class1[0], class2, minkovsky_dist))


def find_nearest_class(classes_list, o1, func_dist_to_class, func_dist_point_to_point) -> int:
    """
    Определяет класс к которому относиться объект, возвращает номер класса
    :param classes_list: список классов
    :param o1: координаты точки
    :param func_dist_to_class: метод вычисления расстояния между точкой и классом
    :param func_dist_point_to_point: метод вычисления расстояния между точками
    :return: номер класса
    """
    list_of_distances = [
        func_dist_to_class(o1, class_, func_dist_point_to_point) for class_ in classes_list
    ]
    enumerated_list = list(enumerate(list_of_distances, 1))
    nearest_dist = min(list_of_distances)
    nearest_class = [n for n, dist in enumerated_list if dist == nearest_dist]
    return nearest_class[0]


fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='r', label='class 1')
ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='b', label='class 2', marker='^')
ax.scatter(class3[:, 0], class3[:, 1], class3[:, 2], c='g', label='class 3', marker='s')
ax.scatter(class4[:, 0], class4[:, 1], class4[:, 2], c='m', label='class 4', marker='d')

try:
    # print("Enter coordinates:\t")
    # x = float(input('x: '))
    # y = float(input('y: '))
    # z = float(input('z: '))
    ax.scatter(MY_X, MY_Y, MY_Z, c='k', label='new point', marker='p')
except ValueError:
    print("\nError")
ax.legend()

# пользовательский ввод методов
while True:
    print("Выберите метод вычисления расстрояния между двумя объектами в двумерном пространстве:")
    print("1. Евклидово расстояния")
    print("2. Расстрояние Минковского")
    objToObjMethod = int(input())
    if objToObjMethod not in [1, 2]:
        continue
    else:
        break

while True:
    print("Выберите метод вычисления расстрояния между объектом и классом:")
    print("1. Расстояние до центроида класса")
    print("2. Найменше з значень відстані до усіх еталонів класу(«найближчий сусід»)")
    objToClassMethod = int(input())
    if objToClassMethod not in [1, 2]:
        continue
    else:
        break

# plt.show()

indices = np.arange(0, 11, 0.5)

fig2 = plt.figure()
ax = fig2.gca(projection='3d')
class_list = [class1, class2, class3, class4]
print()
i = 0
for x in indices:
    for y in indices:
        for z in indices:
            nearest_class = find_nearest_class(class_list, (x, y, z), get_distance_to_centroid, euclid_dist)
            # class_dots[nearest_class].append([x, y, z])
            ax.scatter(x, y, z, c=class_colors[nearest_class], alpha=0.1)
            i += 1
            if (i % 100) == 0:
                print(str(i / ((11 / 0.1) ** 3)), end='\r')
                # print(i)

plt.show()
