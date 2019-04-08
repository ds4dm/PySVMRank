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


def test_help():
    svmrank.help()

def test_alloc_dealloc():
    m = svmrank.Model()

def test_params():
    m = svmrank.Model()

    m.set_params({'-c': 1})
    m._apply_params()

    m.set_params({'-c': -1})
    ok = False
    try:
        m._apply_params()
    except ValueError:
        ok = True
    assert ok

def test_fit():
    m = svmrank.Model({'-c': 1})
    m.fit(train_xs, train_ys, train_groups)

def test_write_read():
    m = svmrank.Model({'-c': 1})
    fd, path = tempfile.mkstemp()
    ok = False
    try:
        m.write(path)
    except ValueError:
        ok = True
    assert ok

    m = svmrank.Model({'-c': 1})
    m.fit(train_xs, train_ys, train_groups)
    fd, path = tempfile.mkstemp()
    m.write(path)

    m = svmrank.Model()
    m.read(path)

def test_predict():
    m = svmrank.Model({'-c': 1})

    ok = False
    try:
        preds = m.predict(test_xs, test_groups)
    except ValueError:
        ok = True
    assert ok

    m.fit(train_xs, train_ys, train_groups)
    preds = m.predict(test_xs, test_groups)

    print(preds)
