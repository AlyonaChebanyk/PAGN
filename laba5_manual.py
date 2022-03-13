import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine

class1 = np.array([[0.05, 0.91],
                   [0.14, 0.96]])
class2 = np.array([[0.49, 0.89],
                   [0.34, 0.81]])

for obj in class1:
    plt.scatter(obj[0], obj[1], c='r', label='class 1')

for obj in class2:
    plt.scatter(obj[0], obj[1], c='b', label='class 2')

x1 = np.linspace(0, 0.5, 4)
y = (x1*5 + 2.16)/3.75
plt.plot(x1, y, c='k')
y = (x1*5 + 2.16 + 1)/3.75
plt.plot(x1, y, c='k', linestyle='--')
y = (x1*5 + 2.16 - 1)/3.75
plt.plot(x1, y, c='k', linestyle='--')

lines = plt.gca().get_lines()
l1 = lines[-3]
l2 = lines[-2]
l3 = lines[-1]
labelLine(l1, 0.25, label='f(x) = 0', align=False, fontsize=12, yoffset=0.05)
labelLine(l2, 0.4, label='f(x)+1 = 0', align=False, fontsize=12, yoffset=0.05)
labelLine(l3, 0.1, label='f(x)-1 = 0', align=False, fontsize=12, yoffset=0.05)

plt.grid(True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('x')
plt.ylabel('y')
plt.show()
