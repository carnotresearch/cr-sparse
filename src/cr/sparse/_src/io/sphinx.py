
import numpy as np

def format_value(value, dtype=None):
    if dtype == 'int64':
        return f'{int(value)}'
    if dtype == np.float64:
        return f'{value:.2f}'
    return f'{value}'
def print_dataframe_as_list_table(df, title):
    print(f'.. list-table:: {title}')
    print('    :header-rows: 1\n')
    cols = df.columns
    dtypes = df.dtypes
    print(f'    * - {df.index.name}')
    for i, col in enumerate(cols):
        print(f'      - {col}')
    for index, row in df.iterrows():
        print(f'    * - {format_value(index)}')
        for i, col in enumerate(cols):
            dtype = dtypes[col]
            print(f'      - {format_value(row[col], dtype)}')
