# Copyright 2021 CR.Sparse Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from jax import jit, lax
import jax.numpy as jnp

from .util import promote_arg_dtypes, check_shapes_are_equal, dtype_ranges

@jit
def mean_squared(array):
    """Returns the mean squared value of an array
    """
    # We need to handle both real and complex cases
    sqr = jnp.conj(array) * array
    mean_sqr = jnp.mean(sqr)
    # Make sure that we are down to float data type
    return jnp.abs(mean_sqr)

@jit
def mean_squared_error(array1, array2):
    """Returns the mean square error between two arrays
    """
    # check shape compatibility
    check_shapes_are_equal(array1, array2)
    # promote to same inexact type (real or complex)
    array1, array2 = promote_arg_dtypes(array1, array2)
    diff = array1 - array2
    # We need to handle both real and complex cases
    sqr_error = jnp.conj(diff) * diff
    mse = jnp.mean(sqr_error)
    # Make sure that we are down to float data type
    return jnp.abs(mse)

def root_mean_squared(array):
    """Returns the root mean squared value of an array
    """
    return jnp.sqrt(mean_squared(array))


def root_mse(array1, array2):
    """Returns the root mean square error between two arrays
    """
    return jnp.sqrt(mean_squared_error(array1, array2))


@partial(jit, static_argnums=(1,))
def normalization_factor(array, normalization : str):
    """Returns the normalization factor based on the contents of an array
    """
    normalization = normalization.lower()
    if normalization == 'euclidean':
        return root_mean_squared(array)
    elif normalization == 'min-max':
        return jnp.abs(jnp.max(array) - jnp.min(array))
    elif normalization == 'mean':
        return jnp.abs(jnp.mean(array))
    elif normalization == 'median':
        return jnp.abs(jnp.median(array))
    else:
        raise ValueError("Unsupported normalization type")


@partial(jit, static_argnames=("normalization",))
def normalized_root_mse(reference_arr, test_arr, normalization='euclidean'):
    """Returns the normalized root mean square error between two arrays
    """
    rmse = root_mse(reference_arr, test_arr)
    denom = normalization_factor(reference_arr, normalization)
    # make sure that denom is non-zero
    eps = jnp.finfo(float).eps
    denom = denom + eps
    return rmse / denom

@jit
def peak_signal_noise_ratio(reference_arr, test_arr):
    """Returns the Peak Signal to Noie Ratio between two arrays 
    """
    min_val, max_val = dtype_ranges[reference_arr.dtype]
    data_min = jnp.min(reference_arr)
    zero = jnp.zeros_like(min_val)
    # min_val below 0 is considered only if the data really has negative values
    min_val = jnp.where(data_min >= 0, zero, min_val)
    drange = max_val - min_val
    mse = mean_squared_error(reference_arr, test_arr)
    eps = jnp.finfo(float).eps
    mse = lax.cond(mse, lambda _ : mse, lambda _ : eps, None)
    return 10 * jnp.log10((drange ** 2) / mse)

@jit
def signal_noise_ratio(reference_arr, test_arr):
    """Returns the signal to noise ratio between a reference array and a test array

    Args:
        reference_arr (jax.numpy.ndarray): Reference signal (can be ND array)
        test_arr (jax.numpy.ndarray): Test signal (can be ND array)

    Returns:
        Signal to Noise Ratio between reference and test signals

    Example:
        ::

            >>> x = jnp.ones(10)
            >>> y = 0.9 * x
            >>> signal_noise_ratio(x, y)
            DeviceArray(19.999998, dtype=float32)
    """
    reference_arr, test_arr = promote_arg_dtypes(reference_arr, test_arr)
    ref_energy = jnp.abs(jnp.vdot(reference_arr, reference_arr))
    error = reference_arr - test_arr
    err_energy = jnp.abs(jnp.vdot(error, error))
    eps = jnp.finfo(float).eps
    # make sure that error energy is non-zero
    err_energy = lax.cond(err_energy, lambda _ : err_energy, lambda _ : eps, None)
    # make sure that ref energy is non-zero
    ref_energy = lax.cond(ref_energy, lambda _ : ref_energy, lambda _ : eps, None)
    return 10 * jnp.log10(ref_energy/ err_energy)

