import jax
import chex
import jax.numpy as jnp

from optax._src.base import GradientTransformation, Updates
from optax._src.combine import chain
from optax._src.transform import update_moment, bias_correction, scale_by_schedule
from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate
from optax._src import numerics

from typing import NamedTuple


class ScaleByAdanState(NamedTuple):
    """State for the Adan algorithm."""
    count: chex.Array
    m: Updates
    v: Updates
    n: Updates
    prev_grad: Updates


def scale_by_adan(
    b1: float = 0.02,
    b2: float = 0.08,
    b3: float = 0.01,
    eps: float = 1e-8,
) -> GradientTransformation:
    """Rescale updates according to the Adan algorithm.
    References:
      [Xie et al, 2022](https://arxiv.org/abs/2208.06677)
    Args:
      b1: the corresponding parameter from the paper.
      b2: the corresponding parameter from the paper.
      b3: the corresponding parameter from the paper.
      eps: term added to the denominator inside the square-root to improve numerical stability.
    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(params):
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        n = jax.tree_util.tree_map(jnp.zeros_like, params)
        prev_grad = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByAdanState(count=jnp.zeros([], jnp.int32), m=m, v=v, n=n, prev_grad=prev_grad)

    def update_fn(updates, state, params=None):
        del params

        decay_m = 1.0 - b1
        decay_v = 1.0 - b2
        decay_n = 1.0 - b3

        m = update_moment(updates, state.m, decay=decay_m, order=1)

        grad_diff = jax.lax.cond(
            pred=state.count == 0,  # Gradient diff is zero on the first iteration
            true_fun=lambda _: jax.tree_util.tree_map(jnp.zeros_like, updates),
            false_fun=lambda _: jax.tree_util.tree_map(lambda cur, prev: cur - prev, updates, state.prev_grad),
            operand=None,
        )

        v = update_moment(grad_diff, state.v, decay=decay_v, order=1)

        second_order_moment_updates = jax.tree_util.tree_map(
            lambda cur, diff: cur + decay_v * diff, updates, grad_diff)
        n = update_moment(second_order_moment_updates, state.n, decay=decay_n, order=2)

        count_inc = numerics.safe_int32_increment(state.count)
        m_hat = bias_correction(m, decay_m, count_inc)
        v_hat = bias_correction(v, decay_v, count_inc)
        n_hat = bias_correction(n, decay_n, count_inc)

        new_updates = jax.tree_util.tree_map(
            lambda mm, vv, nn: (mm + decay_v * vv) / (jnp.sqrt(nn) + eps), m_hat, v_hat, n_hat)

        return new_updates, ScaleByAdanState(count=count_inc, m=m, v=v, n=n, prev_grad=updates)

    return GradientTransformation(init_fn, update_fn)


def adan(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.02,
    b2: float = 0.08,
    b3: float = 0.01,
    eps: float = 1e-8,
) -> GradientTransformation:
    return chain(
        scale_by_adan(b1=b1, b2=b2, b3=b3, eps=eps),
        _scale_by_learning_rate(learning_rate),
    )
