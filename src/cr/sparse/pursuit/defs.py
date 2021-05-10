from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class SingleRecoverySolution:
    signals: jnp.DeviceArray = None
    representations : jnp.DeviceArray = None
    residuals : jnp.DeviceArray =  None
    residual_norms : jnp.DeviceArray = None
    iterations: int = None
    support : jnp.DeviceArray = None

