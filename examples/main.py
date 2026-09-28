import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph_jev_router import JevConditionalEdge

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 1. Initialize the Jev Router for the Multi-Agent System
jev_router = JevConditionalEdge(
    question="Which specialized agent should handle this user request?",
    routes={
        "jira_docs": "JIRA release board monitoring and documentation",
        "analytics": "Data analytics and database querying",
        "support": "General conversational support"
    }
)

# 2. Define dummy nodes for the agents
def jira_agent(state: AgentState):
    return {"messages": ["System One routed to: JIRA Documentation Agent (Derive)"]}

def analytics_agent(state: AgentState):
    return {"messages": ["System One routed to: Data Analytics Agent"]}

def support_agent(state: AgentState):
    return {"messages": ["System One routed to: General Support Agent"]}

# 3. Build the LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("jira_docs", jira_agent)
workflow.add_node("analytics", analytics_agent)
workflow.add_node("support", support_agent)

workflow.set_conditional_entry_point(
    jev_router.route,
    {
        "jira_docs": "jira_docs",
        "analytics": "analytics",
        "support": "support"
    }
)

workflow.add_edge("jira_docs", END)
workflow.add_edge("analytics", END)
workflow.add_edge("support", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Testing Jev Conditional Router...\n")
    
    test_prompts = [
        "Can you generate the release notes for the new JIRA tickets?",
        "Write a SQL query to get the Q3 sales data.",
        "Hello, how do I reset my password?"
    ]
    
    for prompt in test_prompts:
        print(f"User Prompt: {prompt}")
        result = app.invoke({"messages": [prompt]})
        print(f"Result: {result['messages'][-1]}\n")
