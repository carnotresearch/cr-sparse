from .lop_setup import *



@pytest.mark.parametrize("kind,x,x_fwd,x_adj", [
    [
        'regular',
        [1,2,-4, -3, 10, 9, 7, 6, -4, 3, 6, 7.],
        [1., -6., 1., 13., -1., -2., -1., -10., 7., 3., 1., 0.],
        [-1., -1., 6., -1., -13., 1., 2., 1., 10., -7., -3., 6.],
    ],
    [
        'dirichlet',
        [1,2,-4, -3, 10, 9, 7, 6, -4, 3, 6, 7.],
        [ 1., -6., 1., 13., -1., -2., -1., -10., 7., 3., 1., -7.],
        [-1., -1., 6., -1., -13., 1., 2., 1., 10., -7., -3., -1.],
    ],
    [
        'circular',
        [1,2,-4, -3, 10, 9, 7, 6, -4, 3, 6, 7.],
        [ 1., -6., 1., 13., -1., -2., -1., -10., 7., 3., 1., -6.],
        [6., -1., 6., -1., -13., 1., 2., 1., 10., -7., -3., -1.],
    ],
])
def test_tv1d(kind, x, x_fwd, x_adj):
    x = jnp.asarray(x)
    x_fwd = jnp.asarray(x_fwd)
    x_adj = jnp.asarray(x_adj)
    T = lop.tv(len(x), kind)
    T = lop.jit(T)
    assert_allclose(T.times(x), x_fwd)
    assert_allclose(T.trans(x), x_adj)
    A = lop.to_matrix(T)
    AH = lop.to_adjoint_matrix(T)
    assert_allclose(A.T, AH)


@pytest.mark.parametrize("kind,x,x_fwd,x_adj", [
    [
        'regular',
        [[1, 2, 4, 1], [5, 8, 7, 1], [6, 3, 1, 1], [2, 3, 2, 3]],
        [[ 1.+4.j,  2.+6.j, -3.+3.j,  0.+0.j],
        [ 3.+1.j, -1.-5.j, -6.-6.j,  0.+0.j],
        [-3.-4.j, -2.+0.j,  0.+1.j,  0.+2.j],
        [ 1.+0.j, -1.+0.j,  1.+0.j,  0.+0.j]],
        [[-1.-1.j, -1.-2.j, -2.-4.j,  4.-1.j],
        [-5.-4.j, -3.-6.j,  1.-3.j,  7.+0.j],
        [-6.-1.j,  3.+5.j,  2.+6.j,  1.+0.j],
        [-2.+6.j, -1.+3.j,  1.+1.j,  2.+1.j]]
    ],
    [
        'dirichlet',
        [[1, 2, 4, 1], [5, 8, 7, 1], [6, 3, 1, 1], [2, 3, 2, 3]],

        [[ 1.+4.j,  2.+6.j, -3.+3.j, -1.+0.j],
        [ 3.+1.j, -1.-5.j, -6.-6.j, -1.+0.j],
        [-3.-4.j, -2.+0.j,  0.+1.j, -1.+2.j],
        [ 1.-2.j, -1.-3.j,  1.-2.j, -3.-3.j]],
        
        [[-1.-1.j, -1.-2.j, -2.-4.j,  3.-1.j],
        [-5.-4.j, -3.-6.j,  1.-3.j,  6.+0.j],
        [-6.-1.j,  3.+5.j,  2.+6.j,  0.+0.j],
        [-2.+4.j, -1.+0.j,  1.-1.j, -1.-2.j]]
    ],

    [
        'circular',
        [[1, 2, 4, 1], [5, 8, 7, 1], [6, 3, 1, 1], [2, 3, 2, 3]],

        [[ 1.+4.j,  2.+6.j, -3.+3.j,  0.+0.j],
        [ 3.+1.j, -1.-5.j, -6.-6.j,  4.+0.j],
        [-3.-4.j, -2.+0.j,  0.+1.j,  5.+2.j],
        [ 1.-1.j, -1.-1.j,  1.+2.j, -1.-2.j]],

        [[ 0.+1.j, -1.+1.j, -2.-2.j,  3.+2.j],
        [-4.-4.j, -3.-6.j,  1.-3.j,  6.+0.j],
        [-5.-1.j,  3.+5.j,  2.+6.j,  0.+0.j],
        [ 1.+4.j, -1.+0.j,  1.-1.j, -1.-2.j]]
    ],
])
def test_tv2d(kind, x, x_fwd, x_adj):
    x = jnp.asarray(x)
    x_fwd = jnp.asarray(x_fwd)
    x_adj = jnp.asarray(x_adj)
    T = lop.tv2D(x.shape, kind)
    T = lop.jit(T)
    assert_allclose(T.times(x), x_fwd)
    assert_allclose(T.trans(x), x_adj)
    A = lop.to_matrix(T)
    AH = lop.to_adjoint_matrix(T)
    assert_allclose(A.T, AH)
