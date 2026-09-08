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

Upload any CSV or Excel dataset in the Streamlit app. The analyst chooses a
focused backend tool for each question, such as `missing_values`,
`numeric_summary`, `label_distribution`, or `dataset_overview`, based on the
columns and values in the uploaded file. The LangGraph agent in
`backend/agents/graph.py` can call a tool, inspect the result, call another
tool, and then write the final answer.

## Inspect a Dataset from the Command Line

The backend inspection script accepts any supported dataset path:

```text
python -m backend.main path/to/your-dataset.xlsx
```

## Test the Agent

Run a direct agent test by passing the path to the dataset you want to analyze:

```bash
python - <<'PY'
from backend.agents.graph import build_graph

dataset_path = "path/to/your-dataset.xlsx"
result = build_graph().invoke({
	"question": "What is the churn rate and which contract type has the highest churn?",
	"file_path": dataset_path,
	"messages": [],
	"answer": None,
	"tool_name": None,
})
print(result["answer"])
PY
```
