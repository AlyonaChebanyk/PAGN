import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from laba1 import class1, class2, class3, class4
from laba1 import display_classes, get_centroid, standardize
from laba1 import class_colors as cl
from labellines import labelLine, labelLines

classA = np.array([[0.05, 0.91],
                   [0.14, 0.96],
                   [0.16, 0.9],
                   [0.07, 0.7],
                   [0.2, 0,63]])

classB = np.array([[0.49, 0.89],
                   [0.34, 0.81],
                   [0.36, 0.67],
                   [0.47, 0.49],
                   [0.52, 0.53]])


class_colors = {
    '1': 'r',
    '2': 'b',
    '3': 'g',
    '4': 'y'
}




