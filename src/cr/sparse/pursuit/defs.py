from dataclasses import dataclass
import tensorflow as tf

@dataclass
class SingleRecoverySolution:
    signal: tf.Tensor = None
    representation : tf.Tensor = None
    residual : tf.Tensor =  None
    residual_norm : tf.Tensor = None
    iterations: int = None
    support : tf.Tensor = None

