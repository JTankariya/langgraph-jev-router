import os
from typing import Dict
try:
    import typesafe_sdk
except ImportError:
    typesafe_sdk = None

class JevConditionalEdge:
    def __init__(self, question: str, routes: Dict[str, str]):
        self.question = question
        self.routes = routes
        self.api_key = os.getenv("TYPESAFE_API_KEY", "dummy_key")

    def route(self, state: dict) -> str:
        messages = state.get("messages", [])
        if not messages:
            return list(self.routes.keys())[0]
            
        last_message = str(messages[-1])
        
        if typesafe_sdk:
            client = typesafe_sdk.Client(api_key=self.api_key)
            # Map routes to Jev Choice format
            choices = [{"id": k, "description": v} for k, v in self.routes.items()]
            response = client.choice(state=last_message, question=self.question, choices=choices)
            return response.selected_option
        else:
            # Mock behavior for local testing
            lower_msg = last_message.lower()
            if "jira" in lower_msg or "docs" in lower_msg or "release" in lower_msg:
                return "jira_docs"
            elif "sql" in lower_msg or "data" in lower_msg:
                return "analytics"
            return "support"
