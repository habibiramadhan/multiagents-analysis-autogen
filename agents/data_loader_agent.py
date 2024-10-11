from .base_agent import BaseAgent
from tools.data_tools import DataTools
from typing import Dict, Any
import pandas as pd
import json

class DataLoaderAgent(BaseAgent):
    def __init__(self, name: str, llm_config: Dict[str, Any]):
        system_message = """You are a data loading specialist. Your responsibilities include:
        1. Loading CSV files from the input directory
        2. Cleaning and preprocessing the data
        3. Detecting and handling outliers
        4. Encoding categorical variables when necessary"""
        
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.tools = DataTools()
        self.dataframes = {}
        
    def load_and_preprocess(self, input_dir: str = "data/input") -> Dict[str, pd.DataFrame]:
        """Load and preprocess all CSV files."""
        try:
            self.dataframes = self.tools.load_csv_files(input_dir)
            processed_dfs = {}
            for name, df in self.dataframes.items():
                df_cleaned = self.tools.clean_data(df)
                numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
                outliers = self.tools.detect_outliers(df_cleaned, numeric_cols)
                
                categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
                df_processed = self.tools.encode_categorical(df_cleaned, categorical_cols)
                
                processed_dfs[name] = df_processed
                
                output_path = f"data/processed/{name}_processed.csv"
                df_processed.to_csv(output_path, index=False)
                print(f"Saved processed data to: {output_path}")
                
            return processed_dfs
            
        except Exception as e:
            self.handle_error(e)
            return {}