import jax
import jax.numpy as jnp
import optax

from .transform import adan


def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def sphere(x):
    return jnp.sum(x ** 2)


def _test_optimization(func, optimizer, num_iters, initial_params, expected_optimum):
    value_grad_func = jax.value_and_grad(func)
    params = initial_params
    optimizer_state = optimizer.init(params)

    @jax.jit
    def step(params, optimizer_state):
        value, grad = value_grad_func(params)
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return value, params, optimizer_state

    for i in range(num_iters):
        value, params, optimizer_state = step(params, optimizer_state)
        if i % 100 == 0:
            print(f'iter {i}: v={value:.4f} x={params}')

    assert jnp.allclose(params, expected_optimum, atol=1e-3)


def test_adan_sphere():
    return _test_optimization(
        func=sphere,
        optimizer=adan(learning_rate=0.01),
        num_iters=2000,
        initial_params=jnp.array([5.0, -7.5]),
        expected_optimum=jnp.array([0.0, 0.0]),
    )


def test_adan_booth():
    return _test_optimization(
        func=booth,
        optimizer=adan(learning_rate=0.01),
        num_iters=2000,
        initial_params=jnp.array([5.0, -7.5]),
        expected_optimum=jnp.array([1.0, 3.0]),
    )


def test_adam_sphere():
    return _test_optimization(
        func=sphere,
        optimizer=optax.adam(learning_rate=0.01),
        num_iters=3000,
        initial_params=jnp.array([5.0, -7.5]),
        expected_optimum=jnp.array([0.0, 0.0]),
    )


def test_adam_booth():
    return _test_optimization(
        func=booth,
        optimizer=optax.adam(learning_rate=0.01),
        num_iters=5000,
        initial_params=jnp.array([5.0, -7.5]),
        expected_optimum=jnp.array([1.0, 3.0]),
    )
