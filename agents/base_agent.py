import autogen
from typing import Dict, Any, Optional

class BaseAgent(autogen.AssistantAgent):
    def __init__(
        self,
        name: str,
        system_message: str,
        llm_config: Dict[str, Any],
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            **kwargs
        )
        
    def handle_error(self, error: Exception) -> str:
        """Handle errors during agent execution."""
        error_message = f"Error in {self.name}: {str(error)}"
        print(error_message)
        return error_message