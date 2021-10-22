import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
# Denis incoming to project
class_colors = {
    1: 'r',
    2: 'b',
    3: 'g',
    'undefined': 'k'
}

parser = argparse.ArgumentParser()
parser.add_argument('--mi_1', type=float, default=17)
parser.add_argument('--mi_2', type=float, default=19)
parser.add_argument('--mi_3', type=float, default=23)
parser.add_argument('--sigma_1', type=float, default=3)
parser.add_argument('--sigma_2', type=float, default=1.5)
parser.add_argument('--sigma_3', type=float, default=4)
parser.add_argument('--obj_1', type=float, default=17.8)
parser.add_argument('--obj_2', type=float, default=18.3)
parser.add_argument('--obj_3', type=float, default=20)
parser.add_argument('--obj_4', type=float, default=21.1)
parser.add_argument('--p_1', type=float, default=0.1)
parser.add_argument('--p_2', type=float, default=0.8)
parser.add_argument('--p_3', type=float, default=0.1)
parser.add_argument('--r_11', type=int, default=0)
parser.add_argument('--r_12', type=int, default=1)
parser.add_argument('--r_13', type=int, default=3)
parser.add_argument('--r_21', type=int, default=1)
parser.add_argument('--r_22', type=int, default=0)
parser.add_argument('--r_23', type=int, default=2)
parser.add_argument('--r_31', type=int, default=2)
parser.add_argument('--r_32', type=int, default=0)
parser.add_argument('--r_33', type=int, default=4)

namespace = parser.parse_args(sys.argv[1:])

MI_1 = namespace.mi_1
MI_2 = namespace.mi_2
MI_3 = namespace.mi_3
SIGMA_1 = namespace.sigma_1
SIGMA_2 = namespace.sigma_2
SIGMA_3 = namespace.sigma_3
obj_1 = namespace.obj_1
obj_2 = namespace.obj_2
obj_3 = namespace.obj_3
obj_4 = namespace.obj_4
P_1 = namespace.p_1
P_2 = namespace.p_2
P_3 = namespace.p_3
R_11 = namespace.r_11
R_12 = namespace.r_12
R_13 = namespace.r_13
R_21 = namespace.r_21
R_22 = namespace.r_22
R_23 = namespace.r_23
R_31 = namespace.r_31
R_32 = namespace.r_32
R_33 = namespace.r_33


def get_density(mi, sigma, x, p=1):
    """
    Функция вычисляет значение условных плотностей распределения вероятностей
    :param mi: математическое ожидание
    :param sigma: среднеквадратическое отклонение
    :param x: значения координаты х
    :param p: вероятность появления класса
    :return: плотность распределения
    """
    return (1 / sigma * np.sqrt(2 * np.pi)) * np.e ** (-((x - mi) ** 2) / (2 * sigma ** 2)) * p


obj_density = {
    obj_1: [get_density(MI_1, SIGMA_1, obj_1), get_density(MI_2, SIGMA_2, obj_1), get_density(MI_3, SIGMA_3, obj_1)],
    obj_2: [get_density(MI_1, SIGMA_1, obj_2), get_density(MI_2, SIGMA_2, obj_2), get_density(MI_3, SIGMA_3, obj_2)],
    obj_3: [get_density(MI_1, SIGMA_1, obj_3), get_density(MI_2, SIGMA_2, obj_3), get_density(MI_3, SIGMA_3, obj_3)],
    obj_4: [get_density(MI_1, SIGMA_1, obj_4), get_density(MI_2, SIGMA_2, obj_4), get_density(MI_3, SIGMA_3, obj_4)]
}

print(obj_density)

x_indices = np.arange(5, 40, 0.25)
density_1 = get_density(MI_1, SIGMA_1, x_indices)
density_2 = get_density(MI_2, SIGMA_2, x_indices)
density_3 = get_density(MI_3, SIGMA_3, x_indices)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x_indices, density_1, label='class 1', c=class_colors[1])
ax.plot(x_indices, density_2, label='class 2', c=class_colors[2])
ax.plot(x_indices, density_3, label='class 3', c=class_colors[3])

ax.scatter(obj_1, 0, c=class_colors[np.argmax(obj_density[obj_1]) + 1])
ax.plot([obj_1, obj_1], [0, max(obj_density[obj_1])], linestyle='--', c='k')
ax.scatter(obj_2, 0, c=class_colors[np.argmax(obj_density[obj_2]) + 1])
ax.plot([obj_2, obj_2], [0, max(obj_density[obj_2])], linestyle='--', c='k')
ax.scatter(obj_3, 0, c=class_colors[np.argmax(obj_density[obj_3]) + 1])
ax.plot([obj_3, obj_3], [0, max(obj_density[obj_3])], linestyle='--', c='k')
ax.scatter(obj_4, 0, c=class_colors[np.argmax(obj_density[obj_4]) + 1])
ax.plot([obj_4, obj_4], [0, max(obj_density[obj_4])], linestyle='--', c='k')

ax.set_title('Distribution density function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)

fig, ax = plt.subplots(figsize=(10, 5))
density_1 = get_density(MI_1, SIGMA_1, x_indices, P_1)
density_2 = get_density(MI_2, SIGMA_2, x_indices, P_2)
density_3 = get_density(MI_3, SIGMA_3, x_indices, P_3)
ax.plot(x_indices, density_1, label='class 1', c=class_colors[1])
ax.plot(x_indices, density_2, label='class 2', c=class_colors[2])
ax.plot(x_indices, density_3, label='class 3', c=class_colors[3])

obj_density = {
    obj_1: [get_density(MI_1, SIGMA_1, obj_1, P_1), get_density(MI_2, SIGMA_2, obj_1, P_2),
            get_density(MI_3, SIGMA_3, obj_1, P_3)],
    obj_2: [get_density(MI_1, SIGMA_1, obj_2, P_1), get_density(MI_2, SIGMA_2, obj_2, P_2),
            get_density(MI_3, SIGMA_3, obj_2, P_3)],
    obj_3: [get_density(MI_1, SIGMA_1, obj_3, P_1), get_density(MI_2, SIGMA_2, obj_3, P_2),
            get_density(MI_3, SIGMA_3, obj_3, P_3)],
    obj_4: [get_density(MI_1, SIGMA_1, obj_4, P_1), get_density(MI_2, SIGMA_2, obj_4, P_2),
            get_density(MI_3, SIGMA_3, obj_4, P_3)]
}

ax.scatter(obj_1, 0, c=class_colors[np.argmax(obj_density[obj_1]) + 1])
ax.plot([obj_1, obj_1], [0, max(obj_density[obj_1])], linestyle='--', c='k')
ax.scatter(obj_2, 0, c=class_colors[np.argmax(obj_density[obj_2]) + 1])
ax.plot([obj_2, obj_2], [0, max(obj_density[obj_2])], linestyle='--', c='k')
ax.scatter(obj_3, 0, c=class_colors[np.argmax(obj_density[obj_3]) + 1])
ax.plot([obj_3, obj_3], [0, max(obj_density[obj_3])], linestyle='--', c='k')
ax.scatter(obj_4, 0, c=class_colors[np.argmax(obj_density[obj_4]) + 1])
ax.plot([obj_4, obj_4], [0, max(obj_density[obj_4])], linestyle='--', c='k')

ax.set_title('Distribution density function with a priori probability of each class')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)


def get_conditional_risk_charts(r1, r2, r3, x):
    """
    Функция вычисляет условный риск
    :param r1:
    :param r2:
    :param r3:
    :param x:
    :return:
    """
    return get_density(MI_1, SIGMA_1, x, P_1) * r1 + get_density(MI_2, SIGMA_2, x, P_2) * r2 + get_density(MI_3,
                                                                                                           SIGMA_3, x,
                                                                                                           P_3) * r3


fig, ax = plt.subplots(figsize=(10, 5))
risks_1 = get_conditional_risk_charts(R_11, R_21, R_31, x_indices)
risks_2 = get_conditional_risk_charts(R_12, R_22, R_32, x_indices)
risks_3 = get_conditional_risk_charts(R_13, R_23, R_33, x_indices)

ax.plot(x_indices, risks_1, label='class 1', c=class_colors[1])
ax.plot(x_indices, risks_2, label='class 2', c=class_colors[2])
ax.plot(x_indices, risks_3, label='class 3', c=class_colors[3])

obj_risks = {
    obj_1: [get_conditional_risk_charts(R_11, R_21, R_31, obj_1), get_conditional_risk_charts(R_12, R_22, R_32, obj_1),
            get_conditional_risk_charts(R_13, R_23, R_33, obj_1)],
    obj_2: [get_conditional_risk_charts(R_11, R_21, R_31, obj_2), get_conditional_risk_charts(R_12, R_22, R_32, obj_2),
            get_conditional_risk_charts(R_13, R_23, R_33, obj_2)],
    obj_3: [get_conditional_risk_charts(R_11, R_21, R_31, obj_3), get_conditional_risk_charts(R_12, R_22, R_32, obj_3),
            get_conditional_risk_charts(R_13, R_23, R_33, obj_3)],
    obj_4: [get_conditional_risk_charts(R_11, R_21, R_31, obj_3), get_conditional_risk_charts(R_12, R_22, R_32, obj_3),
            get_conditional_risk_charts(R_13, R_23, R_33, obj_3)]
}

ax.scatter(obj_1, 0, c=class_colors[np.argmin(obj_risks[obj_1]) + 1])
ax.plot([obj_1, obj_1], [0, min(obj_risks[obj_1])], linestyle='--', c='k')
ax.scatter(obj_2, 0, c=class_colors[np.argmin(obj_risks[obj_2]) + 1])
ax.plot([obj_2, obj_2], [0, min(obj_risks[obj_2])], linestyle='--', c='k')
ax.scatter(obj_3, 0, c=class_colors[np.argmin(obj_risks[obj_3]) + 1])
ax.plot([obj_3, obj_3], [0, min(obj_risks[obj_3])], linestyle='--', c='k')
ax.scatter(obj_4, 0, c=class_colors[np.argmin(obj_risks[obj_4]) + 1])
ax.plot([obj_4, obj_4], [0, min(obj_risks[obj_4])], linestyle='--', c='k')

ax.set_title('Conditional risk charts')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)

plt.show()
