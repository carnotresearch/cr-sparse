import pytest
import pandas as pd

from cr.sparse import io
import jax.numpy as jnp


def test_print_dataframe_as_list_table():
    d = {
        "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
        "two": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
        "three": pd.Series([1, 2, 3], index=["a", "b", "c"]),
    }
    df = pd.DataFrame(d)
    io.print_dataframe_as_list_table(df, 'abc')