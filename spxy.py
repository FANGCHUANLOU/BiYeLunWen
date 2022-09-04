import numpy as np
def spxy(x, y, test_size=0.2):
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size
    :return: spec_train :(n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    global spec_train, spec_test, target_train, target_test
    x_backup = x
    y_backup = y
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    y = (y - np.mean(y)) / np.std(y)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    for i in range(M - 1):
        xa = x[i, :]
        ya = y[i]
        for j in range((i + 1), M):
            xb = x[j, :]
            yb = y[j]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    target_train = y_backup[m]
    spec_test = x[m_complement, :]
    target_test = y_backup[m_complement]

    return spec_train, spec_test, target_train, target_test
