import numpy as np
from scipy.sparse import csc_matrix


def page_rank(graph, s, err):
    n = float(graph.shape[0])

    matrix = csc_matrix(graph, dtype=np.float)

    col_sum = np.array(matrix.sum(0))[0, :]
    for i in range(col_sum.shape[0]):
        if col_sum[i] != 0:
            for j in range(matrix.indptr[i], matrix.indptr[i + 1]):
                matrix.data[j] /= col_sum[i]

    ro = r = np.ones(n) / n

    loop = True
    iteration_count = 0
    while loop:
        iteration_count += 1

        r = matrix.dot(r) * s + 1 / n * (1 - s)

        for a in (r - ro):
            if abs(a) > err:
                loop = True
                break
            else:
                loop = False

        ro = r

    return r, iteration_count


if __name__ == '__main__':

    a = np.loadtxt('Matrix.txt')
    max_val = int(a.max())

    b = np.zeros([max_val, max_val])

    for i in range(0, len(a)):
        b[a[i][1] - 1][a[i][0] - 1] = a[i][2]
    G = b

    res = page_rank(G, .85, .00001)

    print "\nResult of file \"Matrix.txt\":"
    print "(a) Matrix M: \n", G
    print "(b) The original rank vector (rj): ", np.ones(max_val) / max_val
    print "(c) The Converged rank vector (R): ", res[0]
    print "(d) Iteration = ", res[1]
