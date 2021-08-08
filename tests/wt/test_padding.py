from .setup import *


def test_pad_1d():
    x = [1, 2, 3]
    assert_array_equal(wt.pad(x, (4, 6), 'periodization'),
                       [1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2])
    assert_array_equal(wt.pad(x, (4, 6), 'periodic'),
                       [3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert_array_equal(wt.pad(x, (4, 6), 'constant'),
                       [1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3])
    assert_array_equal(wt.pad(x, (4, 6), 'zero'),
                       [0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0])
    assert_array_equal(wt.pad(x, (4, 6), 'smooth'),
                       [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert_array_equal(wt.pad(x, (4, 6), 'symmetric'),
                       [3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3])
    # assert_array_equal(wt.pad(x, (4, 6), 'antisymmetric'),
    #                    [3, -3, -2, -1, 1, 2, 3, -3, -2, -1, 1, 2, 3])
    assert_array_equal(wt.pad(x, (4, 6), 'reflect'),
                       [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1])
    assert_array_equal(wt.pad(x, (4, 6), 'antireflect'),
                       [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # equivalence of various pad_width formats
    assert_array_equal(wt.pad(x, 4, 'periodic'),
                       wt.pad(x, (4, 4), 'periodic'))

    assert_array_equal(wt.pad(x, (4, ), 'periodic'),
                       wt.pad(x, (4, 4), 'periodic'))

    assert_array_equal(wt.pad(x, [(4, 4)], 'periodic'),
                       wt.pad(x, (4, 4), 'periodic'))


def test_pad_errors():
    # negative pad width
    x = [1, 2, 3]
    assert_raises(ValueError, wt.pad, x, -2, 'periodic')

    # wrong length pad width
    assert_raises(ValueError, wt.pad, x, (1, 1, 1), 'periodic')

    # invalid mode name
    assert_raises(ValueError, wt.pad, x, 2, 'bad_mode')


def test_pad_nd():
    for ndim in [2, 3]:
        x = jnp.arange(4**ndim).reshape((4, ) * ndim)
        if ndim == 2:
            pad_widths = [(2, 1), (2, 3)]
        else:
            pad_widths = [(2, 1), ] * ndim
        for mode in wt.modes:
            xp = wt.pad(x, pad_widths, mode)

            # expected result is the same as applying along axes separably
            xp_expected = x.copy()
            for ax in range(ndim):
                xp_expected = jnp.apply_along_axis(wt.pad,
                                                  ax,
                                                  xp_expected,
                                                  pad_widths=[pad_widths[ax]],
                                                  mode=mode)
            assert_array_equal(xp, xp_expected)
