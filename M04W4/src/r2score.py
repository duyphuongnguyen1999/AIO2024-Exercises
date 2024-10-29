import numpy as np


def r2score(y_pred, y):
    rss = np.sum((y_pred - y) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2


if __name__ == "__main__":
    # Case 1
    y_pred = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    print(r2score(y_pred, y))

    # Case 2
    y_pred = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 5, 2, 4])
    print(r2score(y_pred, y))
