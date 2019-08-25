import numpy as np

from tree.naive_train import train_tree


def test_train_naive_tree():
    x = np.random.normal(size=(1000, 5))
    y = (x.T[0] > 0.1) * 5 + (x.T[2] < 0.01) * 3 + np.random.random(size=(x.shape[0])) * 0.01
    params = {'max_depth': 5, 'gamma': 0.1}
    tree = train_tree.train(x, y, params)
    prediction = tree.predict_many(x)
    assert np.abs(prediction - y).max() < 0.1



