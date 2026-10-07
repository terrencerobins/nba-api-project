import pandas as pd
import os

path = '../data/processed'

for file in os.listdir(path):
    if file.endswith('.parquet'):
        df = pd.read_parquet(f'{path}/{file}')

        print(f'\n--- {file} ({len(df.columns)} columns) ---')

        for col in df.columns:
            print(col)