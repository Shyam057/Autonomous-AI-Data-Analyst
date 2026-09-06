# Autonomous-AI-Data-Analyst

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Backend Script

```bash
python -m backend.main
```

## Run the Streamlit App

```bash
streamlit run app.py
```

To use the same port as the previous local setup:

```bash
streamlit run app.py --server.port 8502
```

The analyst chooses a focused backend tool for each question, such as
`missing_values`, `numeric_summary`, `label_distribution`, or
`dataset_overview`. The LangGraph agent in `backend/agents/graph.py` can call a
tool, inspect the result, call another tool, and then write the final answer.

## Test the Agent

The dataset currently available in this checkout is:

```text
data/Telco_customer_churn.xlsx
```

Run a direct agent test from the project root:

```bash
python - <<'PY'
from backend.agents.graph import build_graph

result = build_graph().invoke({
	"question": "What is the churn rate and which contract type has the highest churn?",
	"file_path": "data/Telco_customer_churn.xlsx",
	"messages": [],
	"answer": None,
	"tool_name": None,
})
print(result["answer"])
PY
```
