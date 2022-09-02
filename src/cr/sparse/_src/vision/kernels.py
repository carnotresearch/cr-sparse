# Copyright 2021 CR-Suite Development Team
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

import jax.numpy as jnp

KERNEL_BOX_3X3 = jnp.ones((3, 3), dtype="float") * (1.0 / (3*3))
"""Simple 3x3 box averaging kernel"""

KERNEL_BOX_5X5 = jnp.ones((5, 5), dtype="float") * (1.0 / (5*5))
"""Simple 5x5 box averaging kernel"""

KERNEL_BOX_7X7 = jnp.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
"""Simple 7x7 box averaging kernel"""

KERNEL_BOX_21X21 = jnp.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
"""Simple 21x21 box averaging kernel"""


KERNEL_SHARPEN_3X3 = jnp.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")
"""3x3 sharpening kernel"""

KERNEL_LAPLACIAN_3X3 = jnp.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")
"""3x3 Laplacian sharpening kernel"""

# construct the Sobel x-axis kernel
KERNEL_SOBEL_X = jnp.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")
"""Horizontal Sobel filter kernel"""

# construct the Sobel y-axis kernel
KERNEL_SOBEL_Y = jnp.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")
"""Vertical Sobel filter kernel"""


def kernel_gaussian(size=5, sigma=1.):
    """Creates a Gaussian kernel of h x w dimensions with the specified sigma
    """
    if isinstance(size, int):
        size = (size, size)
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    # separate out the height and width params
    h, w = size
    # a sequence of points [-l, l] such that total number of points is size
    # for size=3, it is [-1, 0, 1]
    # for size=4, it is [-1.5, -.5, .5, 1.5]
    # for size=5, it is [-2, -1, 0, 1, 2]
    ax_x = jnp.linspace(-(w - 1) / 2., (w - 1) / 2., w)
    ax_y = jnp.linspace(-(h - 1) / 2., (h - 1) / 2., h)
    sigma_x, sigma_y = sigma
    # form a mesh grid of 2 D points
    xx, yy = jnp.meshgrid(ax_x, ax_y)
    # Compute the kernel values at the specified points in the grid
    factor_x = 2 / sigma_x ** 2
    factor_y = 2 / sigma_y ** 2
    kernel = jnp.exp(-(jnp.square(xx) * factor_x + jnp.square(yy) * factor_y))
    # normalize the kernel and return
    return kernel / jnp.sum(kernel)