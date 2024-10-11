# File: main.py

from config.llm_config import get_llm_config
from agents.data_loader_agent import DataLoaderAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.visualization_agent import VisualizationAgent
from agents.reporter_agent import ReporterAgent
from agents.router import AgentRouter
from pathlib import Path
import autogen

def create_output_directories():
    """Create necessary output directories if they don't exist."""
    directories = [
        "data/processed",
        "output/visualizations",
        "output/reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    # Create output directories
    create_output_directories()
    
    # Get LLM configuration
    llm_config = get_llm_config()
    
    # Initialize agents
    router = AgentRouter(
        name="router",
        llm_config=llm_config
    )
    
    data_loader = DataLoaderAgent(
        name="data_loader",
        llm_config=llm_config
    )
    
    analyzer = AnalyzerAgent(
        name="analyzer",
        llm_config=llm_config
    )
    
    visualizer = VisualizationAgent(
        name="visualizer",
        llm_config=llm_config
    )
    
    reporter = ReporterAgent(
        name="reporter",
        llm_config=llm_config
    )
    
    # Create group chat with router
    groupchat = autogen.GroupChat(
        agents=[router, data_loader, analyzer, visualizer, reporter],
        messages=[],
        max_round=50
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    try:
        # Initialize workflow
        router.initialize_workflow()
        
        # Execute workflow
        while True:
            next_task = router.get_next_task()
            if not next_task:
                print("All tasks completed successfully!")
                break
                
            print(f"\nExecuting task: {next_task}")
            
            if next_task == "data_loading":
                # Load and preprocess data
                processed_dfs = data_loader.load_and_preprocess()
                router.route_message(
                    "data_loader",
                    "Data loading completed",
                    {"processed_datasets": processed_dfs}  # Data is now DataFrame
                )
                
            elif next_task == "analysis":
                # Analyze each dataset
                analysis_results = {}
                for dataset_name, df in router.current_state["processed_datasets"].items():
                    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    results = analyzer.analyze_dataset(
                        df=df,
                        analysis_types=["descriptive", "correlation", "regression", "clustering"],
                        feature_columns=numeric_columns
                    )
                    analysis_results[dataset_name] = results
                    
                router.route_message(
                    "analyzer",
                    "Analysis completed",
                    {"analysis_results": analysis_results}
                )
                
            elif next_task == "visualization":
                # Create visualizations for each dataset
                visualization_files = {}
                for dataset_name, df in router.current_state["processed_datasets"].items():
                    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    viz_files = visualizer.create_visualizations(
                        df=df,
                        columns=numeric_columns
                    )
                    visualization_files[dataset_name] = viz_files
                    
                router.route_message(
                    "visualizer",
                    "Visualization completed",
                    {"visualization_files": visualization_files}
                )
                
            elif next_task == "reporting":
                # Generate reports for each dataset
                report_paths = []
                for dataset_name in router.current_state["processed_datasets"].keys():
                    analysis_results = router.current_state["analysis_results"][dataset_name]
                    visualization_files = router.current_state["visualization_files"][dataset_name]
                    
                    for report_type in ["technical", "business"]:
                        report_path = reporter.create_report(
                            analysis_results=analysis_results,
                            visualization_files=visualization_files,
                            report_type=report_type
                        )
                        report_paths.append(report_path)
                        
                router.route_message(
                    "reporter",
                    "Reporting completed",
                    {"report_paths": report_paths}
                )
                
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        recovery_plan = router.create_recovery_plan(e)
        print("\nRecovery plan:")
        print(f"Failed task: {recovery_plan['failed_task']}")
        print("Recovery steps:")
        for step in recovery_plan['recovery_steps']:
            print(f"- {step}")

if __name__ == "__main__":
    main()
