import numpy as np
import matplotlib.pyplot as plt

class1 = np.array([[0, 0],
                   [2, 1]])

class2 = np.array([[1, 0]])


def get_n(class1, class2):
    n_a = len(class1)
    n_b = len(class2)
    n = n_a + n_b
    return n


def get_y(class1, class2):
    y_list = []
    n_a = len(class1)
    n_b = len(class2)
    n = n_a + n_b
    y_list.extend([1] * n_a)
    y_list.extend([-1] * n_b)
    return y_list


def get_kernel(class1, class2):
    n = get_n(class1, class2)
    m = n - 1
    list_of_vectors = np.concatenate([class1, class2])
    scalar_products = np.empty([len(list_of_vectors), len(list_of_vectors)])
    for n1, v1 in enumerate(list_of_vectors):
        for n2, v2 in enumerate(list_of_vectors):
            scalar_products[n1, n2] = (np.dot(v1, v2.T) + 1) ** m
    return scalar_products


def get_A_matrix(class1, class2):
    scalar_products = get_kernel(class1, class2)
    y_list = get_y(class1, class2)
    n = len(y_list)
    A = np.empty([n + 1, n + 1])
    for i in range(n):
        for j in range(n):
            A[i, j] = scalar_products[i, j] * y_list[i] * y_list[j]
        A[i, j + 1] = y_list[i] * (-1)

    A[-1] = y_list + [0]
    return A


def get_lambda_with_any_zero_lambda(class1, class2):
    matrix_a = get_A_matrix(class1, class2)
    y_list = get_y(class1, class2)
    n = len(y_list)

    matrix_b = [[]]
    matrix_b[0].extend([1] * n + [0])
    matrix_b = np.asarray(matrix_b)
    matrix_b = matrix_b.T

    def lam(mat_a, mat_b):
        try:
            result = np.linalg.inv(mat_a).dot(mat_b)
            return result if all(n > 0 for n in result[:-1]) else None
        except:
            return None

    # noinspection PyUnreachableCode
    def get_lambda(old_matrices: dict[tuple: list[np.array, np.array]]) -> dict[tuple: np.array]:
        new_lambda_list = {}
        for indexes, matrices in old_matrices.items():
            old_matrix_a = matrices[0]
            old_matrix_b = matrices[1]
            __lambda: np.ndarray = lam(old_matrix_a, old_matrix_b)
            new_lambda_list.update({indexes: __lambda})
            # if len(__lambda) > 0 and all(n >= 0 for n in __lambda):
            #     for i in reversed(indexes):
            #         __lambda = np.insert(__lambda, i, [0])
            #     return indexes, matrices, __lambda.reshape(len(__lambda), 1)
        return new_lambda_list

    def get_new_matrices(old_matrices: dict[tuple: list[np.array, np.array]]) -> dict[tuple: list[np.array, np.array]]:
        new_matrices = {}
        for indexes, matrices in old_matrices.items():
            assert len(matrices[0]) == len(matrices[1]), "error"
            for i in range(len(matrices[0]) - 1):
                old_matrix_a = matrices[0]
                old_matrix_b = matrices[1]

                new_matrix_a = np.delete(np.delete(old_matrix_a, i, 1), i, 0)
                new_matrix_b = np.delete(old_matrix_b, i, 0)
                key = list(indexes)
                key.pop(i)
                key = tuple(key)
                new_matrices.update({key: [new_matrix_a, new_matrix_b]})
        return new_matrices

    _start_indexes = tuple(range(len(matrix_b)))

    def get_indexes_in_start(key: tuple):
        return tuple(np.unique([key, _start_indexes]))

    def convert_lambda_keys(_lambda: dict[tuple: np.array]):
        _new_lambda_dict = {}
        for keys, v in _lambda.items():
            new_key = []
            for i in range(n):
                if i not in keys:
                    new_key.append(i)
            new_key = tuple(new_key)
            _new_lambda_dict[new_key] = v
        return _new_lambda_dict

    def get_result(old_matrices):
        _lambda = get_lambda(old_matrices)
        result = f_lambda(class1, class2, convert_lambda_keys(_lambda))
        if result is None:
            return get_result(get_new_matrices(old_matrices))
        return result

    _lambda = get_result({_start_indexes: [matrix_a, matrix_b]})
    return _lambda


def f_lambda(class1, class2, lambda_dict: dict):
    A = get_A_matrix(class1, class2)
    B = [[]]
    y_list = get_y(class1, class2)
    n = len(y_list)
    F_values = {}
    for k, v in lambda_dict.items():
        if v is None:
            continue
        lambda_values = np.zeros(n + 1)
        for i in range(n):
            if i not in k:
                lambda_values[i] = v[0]
                v = v[1:]
        lambda_values[-1] = v[-1]
        F_values[tuple(lambda_values)] = get_F(class1, class2, lambda_values)
    if len(F_values) != 0:
        f_max = max(F_values.values())
        for v, f in F_values.items():
            if f == f_max:
                return v
    return None


def get_F(class1, class2, lambda_list):
    scalar_product = get_kernel(class1, class2)
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


def get_f_x(class1, class2, x_list):
    n = get_n(class1, class2)
    y = get_y(class1, class2)
    list_of_vectors = np.concatenate([class1, class2])
    m = n - 1
    kernel = get_kernel(class1, class2)
    _lambda_list = get_lambda_with_any_zero_lambda(class1, class2)
    f_list = []
    for i_x in x_list:
        f_t = 0
        for i_n in range(n):
            f_t += y[i_n] * _lambda_list[i_n] * (
                        list_of_vectors[i_n][0]*i_x + list_of_vectors[i_n][1]*i_x + 1) ** m - \
                   y[i_n] * _lambda_list[i_n] * kernel[0][i_n]
        f_t += 1
        f_list.append(f_t)
    return f_list


# print(get_A_matrix(class1, class2))
x = list(np.arange(-2, 5, 0.1))
y = get_f_x(class1, class2, x)
# y = [-e * e - 2 * e + 1 for e in x]

for obj in class1:
    plt.scatter(obj[0], obj[1], c='r', label='class 1')

for obj in class2:
    plt.scatter(obj[0], obj[1], c='b', label='class 2')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)
plt.grid(True)
plt.show()
