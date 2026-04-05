"""Prompts for experiment log summarization.

Source: treesearch/log_summarization.py
"""

REPORT_SUMMARIZER_SYS_MSG = """You are an expert machine learning researcher.
You are given multiple experiment logs, each representing a node in a stage of exploring scientific ideas and implementations.
Your task is to aggregate these logs and provide scientifically insightful information.

Important instructions:
- Do NOT hallucinate or fabricate information that is not present in the logs.
- Do NOT introduce errors when repeating information from the logs.
- Identify notable insights or differences across the nodes without repeating the same information.
"""

OUTPUT_FORMAT_CONTROL = """Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following fields:
- "Experiment_description": a string describing the conducted experiments
- "Significance": why these experiments are important
- "Description": methods, steps taken, and context
- "List_of_included_plots": list of plots with "path", "description", "analysis"
- "Key_numerical_results": list of results with "result" (float), "description", "analysis"

Ensure the JSON is valid and properly formatted, as it will be automatically parsed."""

REPORT_SUMMARIZER_PROMPT = (
    """You are given multiple experiment logs from different "nodes". Each node represents attempts and experiments exploring various scientific ideas.

The crucial task is to identify the scientific insights gleaned from this stage. Summarize experiments in "Experiment_description", explain processes in "Description", and place key numerical findings in "Key_numerical_results."

Be concise and avoid repeating the same information. Reason carefully about which results are scientifically insightful.

The name of this stage: {stage_name}

Here are the experiment logs:

{node_infos}
"""
    + OUTPUT_FORMAT_CONTROL
)

STAGE_AGGREGATE_PROMPT = """You are given:

1) The summary of all previous experiment stages:
{prev_summary}

2) The name of the current experiment stage:
{stage_name}

3) The summary of the current stage:
{current_summary}

Produce an **updated comprehensive summary** of all experiment stages including new results.

Key Requirements:
1. No Loss of Critical Information — preserve valuable insights, no hallucinations.
2. Merge New Stage Data — integrate new results, remove only clearly redundant content.
3. Maintain insightful plots, figures, and numerical results.
4. The final summary may be very long. That is acceptable.

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```
Ensure the JSON is valid and properly formatted."""

OVERALL_PLAN_SUMMARIZER_PROMPT = """Synthesize a comprehensive summary of the overall plan by integrating details from both the parent and current node plans.
The goal is to create a comprehensive summary of all historical plans, focusing on the main scientific planning and objectives.

Previous overall plan:
{prev_overall_plan}

Current plan:
{current_plan}

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <JSON>, provide:
- "overall_plan": a string that describes the overall plan

Ensure the JSON is valid and properly formatted."""
