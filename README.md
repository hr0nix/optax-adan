# optax-adan
An implementation of adan optimizer for [optax](https://github.com/deepmind/optax/) based on [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models](https://arxiv.org/abs/2208.06677).

Collab with usage example can be found [here](https://colab.research.google.com/drive/19--gju3ELQ9qPbDZbE4NmEnGBLJC901x?usp=sharing).

## How to use:
Install the package:
```bash
python3 -m pip install optax-adan
```

Import the optimizer:
```python3
from optax_adan import adan
```

Use it as you would use any other optimizer from optax:
```python3
# init
optimizer = adan(learning_rate=0.01)
optimizer_state = optimizer.init(initial_params)
# step
grad = grad_func(params)
updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
params = optax.apply_updates(params, updates)
```
