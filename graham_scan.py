import numpy as np


def euclid_dist(o1, o2):
    """
    Функция для нахождения расстояния Эвклида между двумя точками
    :param o1: координаты первой точки
    :param o2: координаты второй точки
    :return: расстояние между точками о1 и о2
    """
    dx = (o1[0] - o2[0]) ** 2
    dy = (o1[1] - o2[1]) ** 2
    d_euclid = np.sqrt(dx + dy)
    return d_euclid


def rotate(A, B, C):
    return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])


def grahamscan(A):
    n = len(A)  # число точек
    P = list(range(n))  # список номеров точек
    for i in range(1, n):
        if A[P[i]][0] < A[P[0]][0]:  # если P[i]-ая точка лежит левее P[0]-ой точки
            P[i], P[0] = P[0], P[i]  # меняем местами номера этих точек
    for i in range(2, n):  # сортировка вставкой
        j = i
        while j > 1 and (rotate(A[P[0]], A[P[j - 1]], A[P[j]]) < 0):
            P[j], P[j - 1] = P[j - 1], P[j]
            j -= 1
    S = [P[0], P[1]]  # создаем стек
    for i in range(2, n):
        while rotate(A[S[-2]], A[S[-1]], A[P[i]]) < 0:
            del S[-1]  # pop(S)
        S.append(P[i])  # push(S,P[i])
    return S


def get_nearest_dots(class1, class2):
    dot1 = class1[0]
    dot2 = class2[0]
    distance = float("inf")
    for o1 in class1:
        for o2 in class2:
            if euclid_dist(o1, o2) < distance:
                distance = euclid_dist(o1, o2)
                dot1 = o1
                dot2 = o2
    return dot1, dot2


def get_three_nearest_dots(class1, class2):
    S1 = grahamscan(class1)
    S2 = grahamscan(class2)
    class1 = [class1[i] for i in S1]
    class2 = [class2[i] for i in S2]
    o1, o2 = get_nearest_dots(class1, class2)
    o1_index = np.where(class1 == o1)[0][0]
    o2_index = np.where(class2 == o2)[0][0]

    if o1_index == len(class1) - 1:
        o1_t_index = 0
        o1_b_index = o1_index - 1
    elif o1_index == 0:
        o1_t_index = o1_index + 1
        o1_b_index = len(class1) - 1
    else:
        o1_t_index = o1_index + 1
        o1_b_index = o1_index - 1

    if o2_index == len(class2) - 1:
        o2_t_index = 0
        o2_b_index = o2_index - 1
    elif o2_index == 0:
        o2_t_index = o2_index + 1
        o2_b_index = len(class2) - 1
    else:
        o2_t_index = o2_index + 1
        o2_b_index = o2_index - 1

    print(o1_index, o1_b_index, o1_t_index)
    print(o2_index, o2_t_index, o2_b_index)

    class1 = [i for n, i in enumerate(class1) if n == o1_index or n == o1_b_index or n == o1_t_index]
    class2 = [i for n, i in enumerate(class2) if n == o2_index or n == o2_b_index or n == o2_t_index]

    return class1, class2

# classA = np.array([[0.05, 0.91],
#                    [0.14, 0.96],
#                    [0.16, 0.9],
#                    [0.07, 0.7],
#                    [0.1, 0.6],
#                    [0.18, 0.89],
#                    [0.1, 0.8],
#                    [0.12, 0.76]])
#
# classB = np.array([[0.49, 0.89],
#                    [0.34, 0.81],
#                    [0.36, 0.67],
#                    [0.47, 0.49],
#                    [0.32, 0.69],
#                    [0.56, 0.42],
#                    [0.5, 0.78],
#                    [0.48, 0.72]])

# S = grahamscan(classA)
#
# ax, fig1 = plt.subplots()
# for obj in classA:
#     plt.scatter(obj[0], obj[1], c='r')
#
# for n, i in enumerate(S):
#     # print([classA[S[n - 1]][0], classA[i][0]], [classA[S[n - 1]][1], classA[i][1]])
#     plt.plot([classA[S[n - 1]][0], classA[i][0]], [classA[S[n - 1]][1], classA[i][1]], c='k')
#
# S = grahamscan(classB)
#
# for obj in classB:
#     plt.scatter(obj[0], obj[1], c='b')
#
# for n, i in enumerate(S):
#     # print([classB[S[n - 1]][0], classB[i][0]], [classB[S[n - 1]][1], classB[i][1]])
#     plt.plot([classB[S[n - 1]][0], classB[i][0]], [classB[S[n - 1]][1], classB[i][1]], c='k')
#
# classA_3, classB_3 = get_three_nearest_dots(classA, classB)
#
# print(classA_3)
# print(classB_3)
# for obj in classA_3:
#     plt.scatter(obj[0], obj[1], c='r', s=[100])
#
# for obj in classB_3:
#     plt.scatter(obj[0], obj[1], c='b', s=[100])
# plt.grid(True)
# plt.show()
