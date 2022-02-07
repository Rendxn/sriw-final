import pandas as pd

columns: list[str] = ['uri', 'name', 'description', 'thumbnail', 'style',
                      'manufacturer', 'class', 'year', 'engine', 'layout']

cat_columns: list[str] = ['style', 'manufacturer', 'class', 'layout']


def content_matrix(data: list) -> tuple[pd.Index, pd.DataFrame]:
    df = pd.DataFrame(data, columns=columns)
    df.set_index(keys=['uri', 'name', 'description',
                 'thumbnail'], inplace=True)
    matrix = pd.get_dummies(df, columns=cat_columns)
    return matrix.index, matrix
