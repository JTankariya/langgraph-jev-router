# LangGraph Jev Router

A native LangGraph conditional edge that replaces slow generative LLMs with TypeSafe AI's deterministic System One model (`Jev`). It enables ultra-fast (sub-500ms), schema-safe routing between specialized agents.

## Why this exists

Standard LangGraph implementations use reasoning models (like Claude or Llama) for conditional edges. This introduces significant latency and the risk of hallucinated routing options. This package uses Jev's parallel `Choice` primitive to route intents deterministically.

## Local Setup & Testing

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

2. **Install the package and its dependencies (LangGraph):**

   ```bash
   pip install -e .
   ```

3. **Run the Example Agentic Handoff:**
   ```bash
   python examples/main.py
   ```
   You will see the router accurately distribute test prompts to the correct mock agents based on intent.
