from .base_agent import BaseAgent
from tools.analysis_tools import AnalysisTools
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import json

class AnalyzerAgent(BaseAgent):
    def __init__(self, name: str, llm_config: Dict[str, Any]):
        system_message = """You are a data analysis specialist. Your responsibilities include:
        1. Performing descriptive statistical analysis
        2. Conducting correlation analysis
        3. Running regression analysis when appropriate
        4. Performing clustering analysis for pattern discovery"""
        
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.tools = AnalysisTools()
        
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Helper function to recursively convert objects to JSON serializable types."""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        else:
            return str(obj)  # Convert any other types to string

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        analysis_types: List[str],
        target_column: str = None,
        feature_columns: List[str] = None
    ) -> Dict[str, Any]:
        """Perform multiple types of analysis on a dataset."""
        try:
            results = {}
            
            # Slice the DataFrame based on feature_columns if provided
            if feature_columns:
                df_features = df[feature_columns]
            else:
                df_features = df  # Use all columns if no feature_columns provided
            
            for analysis_type in analysis_types:
                if analysis_type == "descriptive":
                    stats = self.tools.descriptive_statistics(df_features)
                    results["descriptive_statistics"] = stats
                    
                elif analysis_type == "correlation":
                    corr_matrix = self.tools.correlation_analysis(df_features)
                    results["correlation_analysis"] = corr_matrix.to_dict()  # Convert DataFrame to dict for JSON serialization
                    
                elif analysis_type == "regression" and target_column:
                    reg_results = self.tools.regression_analysis(
                        df, target_column, feature_columns
                    )
                    results["regression_analysis"] = reg_results
                    
                elif analysis_type == "clustering":
                    cluster_results = self.tools.clustering_analysis(
                        df, feature_columns
                    )
                    results["clustering_analysis"] = cluster_results
            
            # Serialize the results to JSON and save to file
            serializable_results = self._convert_to_serializable(results)
            output_path = f"output/analysis_results.json"
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
                
            return serializable_results
            
        except Exception as e:
            self.handle_error(e)
            return {}
