import matplotlib.pyplot as plt
import numpy as np

class1 = np.array([[0.05, 0.91],
                   [0.14, 0.96]])
class2 = np.array([[0.49, 0.89],
                   [0.34, 0.81]])

for obj in class1:
    plt.scatter(obj[0], obj[1], c='r')

for obj in class2:
    plt.scatter(obj[0], obj[1], c='b')

x1 = np.linspace(0.1, 0.4, 4)
y = (x1*5 + 2.16)/3.75
plt.plot(x1, y)
y = (x1*5 + 2.16 + 0.32)/3.75
plt.plot(x1, y)
y = (x1*5 + 2.16 - 0.32)/3.75
plt.plot(x1, y)
plt.grid(True)
plt.show()
