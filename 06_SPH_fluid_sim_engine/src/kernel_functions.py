import jax.numpy as jnp
import jax
import math

def kernel_function_poly6(pts_a, pts_b, h):
    """
    pts_a: [n, 2] array of particle positions
    pts_b: [m, 2] array of particle positions
    h: float, smoothing length
    Returns: [n, ] array of kernel values
    """
    diffs = (pts_a - pts_b)
    mag_squared = jnp.sum(diffs**2, axis=-1)

    # Outside the kernel radius => kernel is zero
    mask = mag_squared > h**2
    mag_squared = jnp.where(mask, 0.0, mag_squared)
    # Inside the kernel radius => kernel is non-zero
    const_2d = 315.0 / (64.0 * math.pi * h**9)
    kernel = const_2d * (h**2 - mag_squared)**3

    # Apply mask to kernel
    ret = jnp.where(mask, 0.0, kernel)
    return ret
    
def kernel_function_gradient_spiky(pts_a, pts_b, h):
    """
    pts_a: [n, 2] array of particle positions
    pts_b: [m, 2] array of particle positions
    h: float, smoothing length
    Returns: [n, 2] array of kernel gradients
    """
    diffs = (pts_a - pts_b)
    mag_squared = jnp.sum(diffs**2, axis=-1)

    # Outside the kernel radius => gradient is zero
    mask = mag_squared > h**2
    mag_squared = jnp.where(mask, 0.0, mag_squared)
    
    # Inside the kernel radius => gradient is non-zero
    mag = jnp.sqrt(mag_squared)
    # Avoid division by zero if particles are at the same spot
    mask_mag_zero = mag == 0.0
    mag = jnp.where(mask_mag_zero, 1.0, mag)

    # 2D Spiky gradient constant
    c = -30.0 / (math.pi * h**5)

    # Multiply by ((h - r)² / r)
    factor = c * (h - mag)**2 / mag

    ret = factor[..., None] * diffs

    # Apply mask to gradient
    ret = jnp.where(mask[..., None], 0.0, ret)
    
    return ret

def kernel_function_viscosity_laplacian(pts_a, pts_b, h):
    """
    pts_a: [n, 2] array of particle positions
    pts_b: [m, 2] array of particle positions
    h: float, smoothing length
    Returns: [n, ] array of kernel viscosity laplacian values
    """
    diffs = (pts_a - pts_b)
    r2 = jnp.sum(diffs**2, axis=-1)

    # Outside the kernel radius => Laplacian is zero
    mask = r2 > h**2
    r2 = jnp.where(mask, 0.0, r2)

    mag = jnp.sqrt(r2)
    
    height_proportion = 1 - (mag / h)
    volume = 3 / (math.pi * h ** 2)
    
    ret = volume * height_proportion

    # Apply mask to laplacian
    ret = jnp.where(mask, 0.0, ret)
    
    return ret