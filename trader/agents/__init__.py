# trader/agents — AI agents that analyze performance data and propose config adjustments.
#
# Design contract:
#   - Agents NEVER write to the DB or call exchange methods directly.
#   - Each agent returns a JSON delta dict.
#   - Python validates the delta against hardcoded guardrail bands before applying.
#   - Guardrails live here in Python, not in prompts — a prompt can be jailbroken,
#     a Python clamp() cannot.
