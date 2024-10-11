from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import autogen
import json
import pandas as pd
import numpy as np

class AgentRouter(BaseAgent):
    def __init__(self, name: str, llm_config: Dict[str, Any]):
        system_message = """You are the router agent responsible for:
        1. Coordinating communication between agents
        2. Managing the flow of tasks
        3. Handling error recovery
        4. Ensuring proper task sequencing
        5. Maintaining the state of the analysis pipeline"""
        
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.task_status = {}
        self.current_state = {}
        
    def initialize_workflow(self) -> Dict[str, Any]:
        """Initialize the workflow state and task status."""
        self.task_status = {
            "data_loading": {"status": "pending", "result": None},
            "analysis": {"status": "pending", "result": None},
            "visualization": {"status": "pending", "result": None},
            "reporting": {"status": "pending", "result": None}
        }
        
        self.current_state = {
            "processed_datasets": {},
            "analysis_results": {},
            "visualization_files": {},
            "report_paths": []
        }
        
        return self.task_status
        
    def update_task_status(
        self,
        task_name: str,
        status: str,
        result: Optional[Any] = None
    ) -> None:
        """Update the status and result of a specific task."""
        self.task_status[task_name] = {
            "status": status,
            "result": result
        }
        
    def get_next_task(self) -> Optional[str]:
        """Determine the next task to be executed based on current state."""
        task_sequence = ["data_loading", "analysis", "visualization", "reporting"]
        
        for task in task_sequence:
            if self.task_status[task]["status"] == "pending":
                return task
                
        return None
        
    def route_message(
        self,
        sender: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route messages between agents and update workflow state."""
        try:
            response = {
                "status": "success",
                "next_task": None,
                "message": "",
                "data": None
            }
            
            # Update state based on message type
            if sender == "data_loader":
                if data and "processed_datasets" in data:
                    self.current_state["processed_datasets"] = data["processed_datasets"]
                    self.update_task_status("data_loading", "completed", data)
                    response["message"] = "Data loading completed successfully"
                    
            elif sender == "analyzer":
                if data and "analysis_results" in data:
                    self.current_state["analysis_results"] = data["analysis_results"]
                    self.update_task_status("analysis", "completed", data)
                    response["message"] = "Analysis completed successfully"
                    
            elif sender == "visualizer":
                if data and "visualization_files" in data:
                    self.current_state["visualization_files"] = data["visualization_files"]
                    self.update_task_status("visualization", "completed", data)
                    response["message"] = "Visualization completed successfully"
                    
            elif sender == "reporter":
                if data and "report_paths" in data:
                    self.current_state["report_paths"].extend(data["report_paths"])
                    self.update_task_status("reporting", "completed", data)
                    response["message"] = "Reporting completed successfully"
            
            # Determine next task
            next_task = self.get_next_task()
            response["next_task"] = next_task
            
            # Save current state
            self._save_state()
            
            return response
            
        except Exception as e:
            error_msg = self.handle_error(e)
            return {
                "status": "error",
                "next_task": None,
                "message": error_msg,
                "data": None
            }
            
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

    def _save_state(self) -> None:
        """Save current state to file for recovery purposes."""
        state = {
            "task_status": self._convert_to_serializable(self.task_status),
            "current_state": self._convert_to_serializable(self.current_state)
        }
        
        with open("output/workflow_state.json", "w") as f:
            json.dump(state, f, indent=4)
            
    def load_state(self) -> bool:
        """Load previous state if available."""
        try:
            with open("output/workflow_state.json", "r") as f:
                state = json.load(f)
                self.task_status = state["task_status"]
                
                # Manually convert back to DataFrames if necessary
                if "processed_datasets" in state["current_state"]:
                    self.current_state["processed_datasets"] = {
                        key: pd.DataFrame(value) for key, value in state["current_state"]["processed_datasets"].items()
                    }
                self.current_state["analysis_results"] = state["current_state"]["analysis_results"]
                self.current_state["visualization_files"] = state["current_state"]["visualization_files"]
                self.current_state["report_paths"] = state["current_state"]["report_paths"]
            return True
        except FileNotFoundError:
            return False
            
    def create_recovery_plan(self, error: Exception) -> Dict[str, Any]:
        """Create a recovery plan when an error occurs."""
        recovery_plan = {
            "error": str(error),
            "failed_task": None,
            "recovery_steps": []
        }
        
        # Identify failed task
        for task, status in self.task_status.items():
            if status["status"] == "pending":
                recovery_plan["failed_task"] = task
                break
                
        # Create recovery steps
        if recovery_plan["failed_task"] == "data_loading":
            recovery_plan["recovery_steps"] = [
                "Verify input data exists and is accessible",
                "Check CSV file format and encoding",
                "Attempt to load individual files separately",
                "Skip problematic files if necessary"
            ]
        elif recovery_plan["failed_task"] == "analysis":
            recovery_plan["recovery_steps"] = [
                "Verify data types are appropriate for analysis",
                "Handle missing values or outliers",
                "Reduce feature set if necessary",
                "Try alternative analysis methods"
            ]
        elif recovery_plan["failed_task"] == "visualization":
            recovery_plan["recovery_steps"] = [
                "Check if data is appropriate for visualization",
                "Reduce dataset size if necessary",
                "Try alternative visualization types",
                "Skip problematic visualizations"
            ]
        elif recovery_plan["failed_task"] == "reporting":
            recovery_plan["recovery_steps"] = [
                "Verify all required results are available",
                "Check file permissions for report output",
                "Try alternative report format",
                "Generate partial report if necessary"
            ]
            
        return recovery_plan
