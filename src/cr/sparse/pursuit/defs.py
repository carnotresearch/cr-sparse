from dataclasses import dataclass
import tensorflow as tf

@dataclass
class SingleRecoverySolution:
    signals: tf.Tensor = None
    representations : tf.Tensor = None
    residuals : tf.Tensor =  None
    residual_norms : tf.Tensor = None
    iterations: int = None
    support : tf.Tensor = None

