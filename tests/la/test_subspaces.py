from .setup import *

# ambient dimension
n = 10
# subspace dimension
d = 4
# number of subspaces
k = 2

bases = crdata.random_subspaces_jit(crs.KEYS[0], n, d, k)


def test_orth_complement():
    A = bases[0]
    B = bases[1]
    C = subspaces.orth_complement(A, B)
    G = A.T @ C
    # all vectors in C must be orthogonal to vectors in A
    assert_allclose(G, 0, atol=atol)


def test_principal_angles():
    A = bases[0]
    B = bases[1]
    C = subspaces.orth_complement(A, B)
    angles = subspaces.principal_angles_cos_jit(A, C)
    # all vectors in C must be orthogonal to vectors in A
    assert_allclose(angles, 0, atol=atol)
    angles = subspaces.principal_angles_rad_jit(A, C)
    assert_allclose(angles, jnp.pi/2, atol=atol)
    angles = subspaces.principal_angles_deg_jit(A, C)
    assert_allclose(angles, 90, atol=atol)

def test_smallest_principal_angle():
    A = bases[0]
    B = bases[1]
    C = subspaces.orth_complement(A, B)
    angle = subspaces.smallest_principal_angle_cos_jit(A, C)
    assert_allclose(angle, 0, atol=atol)
    angle = subspaces.smallest_principal_angle_rad_jit(A, C)
    assert_allclose(angle, jnp.pi/2, atol=atol)
    angle = subspaces.smallest_principal_angle_deg_jit(A, C)
    assert_allclose(angle, 90, atol=atol)


def test_smallest_principal_angles():
    A = bases[0]
    lst = jnp.stack((A, A, A))
    angles = subspaces.smallest_principal_angles_cos_jit(lst)
    o = jnp.ones((3,3))
    z = jnp.zeros((3,3))
    assert_allclose(o, angles)
    angles = subspaces.smallest_principal_angles_rad_jit(lst)
    # allow for 32-bit floating point errors
    assert_allclose(z, angles, atol=1e-1)
    angles = subspaces.smallest_principal_angles_deg_jit(lst)
    # allow for 32-bit floating point errors
    assert_allclose(z, angles, atol=1e-1)


def test_project_to_subspace():
    A  = jnp.eye(6)[:, :3]
    v = jnp.arange(6) + 0.
    u = subspaces.project_to_subspace(A, v)
    u0 = jnp.array([0., 1., 2., 0., 0., 0.])
    assert_allclose(u, u0)

def test_is_in_subspace():
    A  = jnp.eye(6)[:, :3]
    v = jnp.arange(6) + 0.
    u = subspaces.project_to_subspace(A, v)
    assert subspaces.is_in_subspace(A, u)
    assert not subspaces.is_in_subspace(A, v)

def test_subspace_distance():
    A  = jnp.eye(6)[:, :3]
    d = subspaces.subspace_distance(A, A)
    assert_array_equal(d, 0)
