from .base_agent import BaseAgent
from tools.visualization_tools import VisualizationTools
from typing import Dict, Any, List
import pandas as pd

class VisualizationAgent(BaseAgent):
    def __init__(self, name: str, llm_config: Dict[str, Any]):
        system_message = """You are a data visualization specialist. Your responsibilities include:
        1. Creating appropriate visualizations based on data types
        2. Generating both static and interactive plots
        3. Ensuring visualizations are clear and informative
        4. Saving visualizations in appropriate formats"""
        
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.tools = VisualizationTools()
        
    def create_visualizations(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_column: str = None
    ) -> Dict[str, List[str]]:
        """Create a suite of visualizations for the dataset."""
        try:
            visualization_files = {
                'static': [],
                'interactive': []
            }
            
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            datetime_columns = df.select_dtypes(include=['datetime64']).columns
            
            # Pairplot for numeric columns
            if len(numeric_columns) > 1:
                pairplot_file = self.tools.create_static_plot(
                    df,
                    'pairplot',
                    x_column=numeric_columns[0],
                    columns=numeric_columns,
                    title='Pairplot of Numeric Variables'
                )
                visualization_files['static'].append(pairplot_file)
            
            # Correlation heatmap for numeric columns
            if len(numeric_columns) > 1:
                heatmap_file = self.tools.create_static_plot(
                    df[numeric_columns],
                    'heatmap',
                    x_column=numeric_columns[0],
                    title='Correlation Heatmap'
                )
                visualization_files['static'].append(heatmap_file)
            
            for col in columns:
                if col in numeric_columns:
                    # Histogram
                    hist_file = self.tools.create_static_plot(
                        df,
                        'histogram',
                        x_column=col,
                        title=f'Distribution of {col}'
                    )
                    visualization_files['static'].append(hist_file)
                    
                    # Box plot
                    if target_column and target_column in categorical_columns:
                        box_file = self.tools.create_static_plot(
                            df,
                            'boxplot',
                            x_column=target_column,
                            y_column=col,
                            title=f'Box Plot of {col} by {target_column}'
                        )
                        visualization_files['static'].append(box_file)
                    
                    # Interactive scatter plot
                    if target_column and target_column in numeric_columns:
                        scatter_file = self.tools.create_interactive_plot(
                            df,
                            'scatter',
                            x_column=col,
                            y_column=target_column,
                            title=f'{col} vs {target_column}'
                        )
                        visualization_files['interactive'].append(scatter_file)
                
                elif col in categorical_columns:
                    # Bar plot
                    bar_file = self.tools.create_static_plot(
                        df,
                        'bar',
                        x_column=col,
                        title=f'Distribution of {col}'
                    )
                    visualization_files['static'].append(bar_file)
                
                elif col in datetime_columns:
                    # Time series plot
                    if target_column and target_column in numeric_columns:
                        time_series_file = self.tools.create_interactive_plot(
                            df.sort_values(col),
                            'line',
                            x_column=col,
                            y_column=target_column,
                            title=f'{target_column} over Time'
                        )
                        visualization_files['interactive'].append(time_series_file)
            
            return visualization_files
            
        except Exception as e:
            self.handle_error(e)
            return {'static': [], 'interactive': []}