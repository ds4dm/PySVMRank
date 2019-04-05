import numpy as np
import tempfile
import svmrank


train_xs = np.array([
    [1, 1, 0, 0.2, 0],
    [0, 0, 1, 0.1, 1],
    [0, 1, 0, 0.4, 0],
    [0, 0, 1, 0.3, 0],
    [0, 0, 1, 0.2, 0],
    [1, 0, 1, 0.4, 0],
    [0, 0, 1, 0.1, 0],
    [0, 0, 1, 0.2, 0],
    [0, 0, 1, 0.1, 1],
    [1, 1, 0, 0.3, 0],
    [1, 0, 0, 0.4, 1],
    [0, 1, 1, 0.5, 0],
])
train_ys = np.array([
    3,
    2,
    1,
    1,
    1,
    2,
    1,
    1,
    2,
    3,
    4,
    1,
])
train_groups = np.array([
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
])

test_xs = np.array([
    [1, 0, 0, 0.2, 1],
    [1, 1, 0, 0.3, 0],
    [0, 0, 0, 0.2, 1],
    [0, 0, 1, 0.2, 0],
])
test_ys = np.array([
    4,
    3,
    2,
    1,
])
test_groups = np.array([
    4,
    4,
    4,
    4,
])


def test_alloc_dealloc():
    m = svmrank.Model()


def test_fit():
    m = svmrank.Model()
    m.fit(train_xs, train_ys, train_groups, params={
        '-c': 1,
    })

def test_write():
    m = svmrank.Model()
    m.fit(train_xs, train_ys, train_groups, params={
        '-c': 1,
    })
    fd, path = tempfile.mkstemp()
    m.write(path)
    print(f"model written to {path}")

