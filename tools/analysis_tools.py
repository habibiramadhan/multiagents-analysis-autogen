import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
from sklearn.linear_model import LinearRegression

class AnalysisTools:
    @staticmethod
    def descriptive_statistics(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Calculate descriptive statistics for specified columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                }
        
        return stats

    @staticmethod
    def correlation_analysis(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Calculate correlation matrix for specified columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        return df[columns].corr()

    @staticmethod
    def regression_analysis(df: pd.DataFrame, target: str, features: List[str]) -> Dict[str, Any]:
        """Perform linear regression analysis."""
        X = df[features]
        y = df[target]
        
        model = LinearRegression()
        model.fit(X, y)
        
        results = {
            'coefficients': dict(zip(features, model.coef_)),
            'intercept': model.intercept_,
            'r_squared': model.score(X, y)
        }
        
        return results

    @staticmethod
    def clustering_analysis(df: pd.DataFrame, columns: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """Perform K-means clustering analysis."""
        # Ensure that columns provided are from the DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a DataFrame for clustering analysis.")
        
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()  # Default to numeric columns
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[columns])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        results = {
            'clusters': clusters.tolist(),
            'cluster_centers': {f'cluster_{i}': dict(zip(columns, centers)) for i, centers in enumerate(cluster_centers)},
            'inertia': kmeans.inertia_
        }
        
        return results
