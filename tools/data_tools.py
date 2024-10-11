import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path

class DataTools:
    @staticmethod
    def load_csv_files(input_dir: str) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the input directory."""
        csv_files = list(Path(input_dir).glob("*.csv"))
        dataframes = {}
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                dataframes[file_path.stem] = df
                print(f"Successfully loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        if not dataframes:
            print(f"No CSV files found in {input_dir}")
        
        return dataframes

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe by handling missing values and outliers."""
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill missing values
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates()
        
        print(f"Cleaned data: {len(df) - len(df_cleaned)} duplicate rows removed")
        
        return df_cleaned

    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[int]]:
        """Detect outliers using IQR method."""
        outliers = {}
        
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_indices = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].index.tolist()
                outliers[col] = outlier_indices
        
        print(f"Outliers detected in {len(outliers)} columns")
        
        return outliers

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables."""
        df_encoded = df.copy()
        encoded_columns = []
        
        for col in columns:
            if df_encoded[col].dtype == 'object':
                df_encoded[f"{col}_encoded"] = pd.factorize(df_encoded[col])[0]
                encoded_columns.append(f"{col}_encoded")
        
        print(f"Encoded {len(encoded_columns)} categorical columns")
        
        return df_encoded