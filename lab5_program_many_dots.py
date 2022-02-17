import matplotlib.pyplot as plt
import numpy as np

classA = np.array([[0.05, 0.91],
                   [0.14, 0.96],
                   [0.16, 0.9],
                   [0.07, 0.7]])

classB = np.array([[0.49, 0.89],
                   [0.34, 0.81],
                   [0.36, 0.67],
                   [0.47, 0.49]])


def get_scalar_product(class1, class2):
    list_of_vectors = np.concatenate([class1, class2])
    scalar_products = np.empty([len(list_of_vectors), len(list_of_vectors)])
    for n1, v1 in enumerate(list_of_vectors):
        for n2, v2 in enumerate(list_of_vectors):
            scalar_products[n1, n2] = np.dot(v1, v2.T)
    return scalar_products


def get_y(class1, class2):
    y_list = []
    n_a = len(class1)
    n_b = len(class2)
    n = n_a + n_b
    y_list.extend([1] * n_a)
    y_list.extend([-1] * n_b)
    return y_list


def get_n(class1, class2):
    n_a = len(class1)
    n_b = len(class2)
    n = n_a + n_b
    return n


def get_A_matrix(class1, class2):
    scalar_products = get_scalar_product(class1, class2)
    y_list = get_y(class1, class2)
    n = len(y_list)
    A = np.empty([n + 1, n + 1])
    for i in range(n):
        for j in range(n):
            A[i, j] = scalar_products[i, j] * y_list[i] * y_list[j]
        A[i, j + 1] = y_list[i] * (-1)

    A[-1] = y_list + [0]
    return A


def get_lambda_with_zero_zero_lambda(class1, class2):
    A = get_A_matrix(class1, class2)
    B = [[]]
    n = get_n(class1, class2)
    B[0].extend([1] * n + [0])
    B = np.asarray(B)
    B = B.T
    lambda_ = np.linalg.inv(A).dot(B)
    return lambda_


def get_lambda_with_one_zero_lambda(class1, class2):
    A = get_A_matrix(class1, class2)
    B = [[]]
    n = get_n(class1, class2)
    lambda_list_one_zero = []
    B[0].extend([1] * (n - 1) + [0])
    B = np.asarray(B)
    B = B.T
    for i in range(n):
        new_A = np.delete(np.delete(A, i, 1), i, 0)
        lambda_list_one_zero.append(np.linalg.inv(new_A).dot(B))
    for i in lambda_list_one_zero:
        print(i)


def get_lambda_with_any_zero_lambda(class1, class2, t=0):
    matrix_a = get_A_matrix(class1, class2)
    y_list = get_y(class1, class2)
    n = len(y_list)

    matrix_b = [[]]
    matrix_b[0].extend([1] * (n - t) + [0])
    matrix_b = np.asarray(matrix_b)
    matrix_b = matrix_b.T

    def lam(mat_a, mat_b):
        try:
            result = np.linalg.inv(mat_a).dot(mat_b)
        except:
            result = []
        return result

    # noinspection PyUnreachableCode
    def get_lambda(old_matrices: dict[tuple: list[np.array, np.array]]) -> tuple[
        str, dict[tuple: list[np.array, np.array]], np.array]:
        for indexes, matrices in old_matrices.items():
            old_matrix_a = matrices[0]
            old_matrix_b = matrices[1]
            __lambda = lam(old_matrix_a, old_matrix_b)
            if len(__lambda) > 0 and all(n >= 0 for n in __lambda):
                return indexes, matrices, __lambda
        new_matrices = {}
        for indexes, matrices in old_matrices.items():
            assert len(matrices[0]) == len(matrices[1]), "error"
            for i in range(len(matrices[0])):
                old_matrix_a = matrices[0]
                old_matrix_b = matrices[1]

                new_matrix_a = np.delete(np.delete(old_matrix_a, i, 1), i, 0)
                new_matrix_b = np.delete(old_matrix_b, i, 0)
                key: list = sorted(indexes)
                key.append(i)
                key: tuple = tuple(key)
                if key not in indexes:
                    new_matrices.update({key: [new_matrix_a, new_matrix_b]})
        return get_lambda(new_matrices)
# (0, 3, 4, 8)
# (0, 2, 2, 4)
    return get_lambda({(): [matrix_a, matrix_b]})


def get_lambda_with_two_zero_lambda(class1, class2):
    A = get_A_matrix(class1, class2)
    B = [[]]
    y_list = get_y(class1, class2)
    n = len(y_list)
    B[0].extend([1] * (n - 2) + [0])
    B = np.asarray(B)
    B = B.T
    lambda_dict_two_zero = {}
    for i in range(n):
        for j in range(n):
            if i != j and y_list[i] * y_list[j] == -1:
                print(i, j)
                new_A = np.delete(np.delete(A, [i, j], 1), [i, j], 0)
                try:
                    lambda_dict_two_zero[frozenset([i, j])] = np.linalg.inv(new_A).dot(B)
                except np.linalg.LinAlgError:
                    lambda_dict_two_zero[frozenset([i, j])] = None

    F_values = {}
    for k, v in lambda_dict_two_zero.items():
        if v is None:
            continue
        lambda_values = np.zeros(n + 1)
        for i in range(n):
            if i not in k:
                lambda_values[i] = v[0]
        lambda_values[-1] = v[-1]
        F_values[tuple(lambda_values)] = get_F(class1, class2, lambda_values)
    f_max = max(F_values.values())
    for v, f in F_values.items():
        if f == f_max:
            return v


def get_F(class1, class2, lambda_list):
    scalar_product = get_scalar_product(class1, class2)
    y_list = get_y(class1, class2)
    n = get_n(class1, class2)
    F = 0
    F += sum(lambda_list[0: -1])
    for i in range(n):
        for j in range(n):
            F += -1 / 2 * (lambda_list[i] * lambda_list[j] * y_list[i] * y_list[j] * scalar_product[i, j])
    for i in range(n):
        F += lambda_list[-1] * lambda_list[i] * y_list[i]
    return F


def get_w_b(class1, class2, lambda_list):
    class_list = np.concatenate([class1, class2])
    y_list = get_y(class1, class2)
    n = len(y_list)
    w = [0, 0]
    print(lambda_list)
    for i in range(n):
        w += lambda_list[i] * y_list[i] * class_list[i]

    b_index = 0
    for n, l in enumerate(lambda_list):
        if l > 0:
            b_index = n
            break
    # b_list = [y_list[b_index]**-1 - np.asarray(w).dot(class_list[b_index]) for b_index in range(n)]
    # b = np.mean(b_list)
    b = y_list[b_index] ** -1 - np.asarray(w).dot(class_list[b_index])
    # b = 0
    return w, b


def get_h(w):
    return 2 / (np.sqrt(w[0] ** 2 + w[1] ** 2))


if __name__ == '__main__':
    keys, matrixes, lambda_list = get_lambda_with_any_zero_lambda(classA, classB)

    x1 = np.linspace(0, 0.5, 4)

    ax, fig1 = plt.subplots()
    for obj in classA:
        plt.scatter(obj[0], obj[1], c='r')

    for obj in classB:
        plt.scatter(obj[0], obj[1], c='b')
    w, b = get_w_b(classA, classB, lambda_list)
    print(get_h(w))
    h = get_h(w)
    y = - (x1 * w[0] + b) / w[1]
    plt.plot(x1, y)
    y = - (x1 * w[0] + b + 1) / w[1]
    plt.plot(x1, y)
    y = - (x1 * w[0] + b - 1) / w[1]
    plt.grid(True)
    plt.plot(x1, y)

    plt.show()
