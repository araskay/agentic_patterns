'''
loads kaggle bike store data set into a duckdb
to provide a quick way to play with the agent.
'''

import os
import duckdb
import kagglehub
from sqlalchemy import create_engine

class BikeStoreDb:
    def __init__(self, db_path='bike_store.db'):
        self.db_path = db_path
        self.download_path = self._download_data()
        self._create_db()

    @staticmethod
    def _download_data():
        '''
        Download the latest version of the bike store dataset
        '''
        path = kagglehub.dataset_download("dillonmyrick/bike-store-sample-database")
        print(f"Downloaded dataset to {path}")
        return path
    
    def _create_db(self):
        con = duckdb.connect(database=self.db_path, read_only=False)
        csv_count = 0
        for fname in os.listdir(self.download_path):
            fpath = os.path.join(self.download_path, fname)
            if fname.lower().endswith(".csv"):
                csv_count += 1
                # Create a table name from the filename (without extension)
                table_name = os.path.splitext(fname)[0]
                
                # Create a DuckDB table by reading the CSV with automatic detection
                con.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} AS
                    SELECT * FROM read_csv_auto('{fpath}');
                    """
                )
        con.close()
        print(f"Created {csv_count} tables in {self.db_path}")  

    def get_engine(self):
        engine = create_engine(f"duckdb:///{self.db_path}")
        return engine

