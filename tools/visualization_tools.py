import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

class VisualizationTools:
    @staticmethod
    def create_static_plot(
        df: pd.DataFrame,
        plot_type: str,
        x_column: str,
        y_column: str = None,
        title: str = None,
        **kwargs
    ) -> str:
        """Create various types of static plots and save them to file."""
        plt.figure(figsize=(12, 6))
        
        if plot_type == "scatter":
            sns.scatterplot(data=df, x=x_column, y=y_column)
        elif plot_type == "line":
            sns.lineplot(data=df, x=x_column, y=y_column)
        elif plot_type == "bar":
            if y_column:
                sns.barplot(data=df, x=x_column, y=y_column)
            else:
                df[x_column].value_counts().plot(kind='bar')
        elif plot_type == "histogram":
            sns.histplot(data=df, x=x_column, kde=True)
        elif plot_type == "boxplot":
            sns.boxplot(data=df, x=x_column, y=y_column)
        elif plot_type == "heatmap":
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        elif plot_type == "pairplot":
            sns.pairplot(df[kwargs.get('columns', df.columns)])
            plt.tight_layout()
        
        if title:
            plt.title(title)
        
        os.makedirs("output/visualizations", exist_ok=True)
        filename = f"output/visualizations/static_{plot_type}_{x_column}"
        if y_column:
            filename += f"_{y_column}"
        filename += ".png"
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        return filename

    @staticmethod
    def create_interactive_plot(
        df: pd.DataFrame,
        plot_type: str,
        x_column: str,
        y_column: str = None,
        title: str = None,
        **kwargs
    ) -> str:
        """Create interactive plots using plotly."""
        if plot_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_column, title=title)
        elif plot_type == "line":
            fig = px.line(df, x=x_column, y=y_column, title=title)
        elif plot_type == "bar":
            if y_column:
                fig = px.bar(df, x=x_column, y=y_column, title=title)
            else:
                fig = px.bar(df[x_column].value_counts().reset_index(), x='index', y=x_column, title=title)
        elif plot_type == "histogram":
            fig = px.histogram(df, x=x_column, title=title)
        elif plot_type == "box":
            fig = px.box(df, x=x_column, y=y_column, title=title)
        elif plot_type == "heatmap":
            fig = px.imshow(df.corr(), title=title)
        elif plot_type == "pairplot":
            fig = px.scatter_matrix(df[kwargs.get('columns', df.columns)], title=title)
        
        os.makedirs("output/visualizations", exist_ok=True)
        filename = f"output/visualizations/interactive_{plot_type}_{x_column}"
        if y_column:
            filename += f"_{y_column}"
        filename += ".html"
        
        fig.write_html(filename)
        
        return filename