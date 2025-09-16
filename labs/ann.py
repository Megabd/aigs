# %% Imports
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from jax import grad

# %%
mnist = fetch_openml("mnist_784", version=1)
X = jnp.array(mnist.data.astype("float32"))  # shape (70000, 784)
y = jnp.array(np.eye(10)[mnist.target.astype("int64")])  # shape (70000,)


def apply_fn(params, X):
    return relu(relu(X @ params["W_1"]) @ params["W_2"])


def init_fn(cfg):
    return {
        "W_1": jnp.array(np.random.randn(X.shape[1], cfg.hidden)) * 0.01,
        "W_2": jnp.array(np.random.randn(cfg.hidden, 10)) * 0.01,
    }


def main(cfg):
    params = init_fn(cfg)
    grad_fn = grad(loss_fn)

    # model output
    for epoch in range(cfg.epochs):
        grads = grad_fn(params, X, y)
        sns.heatmap(np.array(grads["W_2"]))
        plt.show()
        params = {"W_1": params["W_1"] - cfg.lr * grads["W_1"], "W_2": params["W_2"] - cfg.lr * grads["W_2"]}
        loss = loss_fn(params, X, y)
        print(f"Epoch {epoch}: Loss = {loss}, Accuracy = {acc(y, apply_fn(params, X))}")

    return params


def loss_fn(params, X, y):
    y_hat = apply_fn(params, X)
    return ((y - y_hat) ** 2).mean()


def relu(x):
    return jnp.maximum(0, x)


def acc(y, y_hat):
    return (jnp.argmax(y, axis=1) == jnp.argmax(y_hat, axis=1)).mean()
