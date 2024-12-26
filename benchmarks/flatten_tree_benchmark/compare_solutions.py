import pandas as pd

def main():
    pandas_df = pd.read_parquet('data/tree_updates_hierarchy_pandas.parquet')
    duckdb_df = pd.read_parquet('data/tree_updates_hierarchy_duckdb.parquet')

    ...
if __name__ == '__main__':
    main()