import inspect
import os

from langchain_core.messages import AIMessage

os.environ["LANGSMITH_TRACING"] = "false"

from recruiting_agent import recruiting_agent as agent_module


EXPECTED_TOOLS = {
    "lookup_job_posting",
    "build_candidate_profile",
    "get_candidate",
    "send_candidate_email",
    "score_candidate",
    "add_candidate_skill",
    "get_current_recruiter",
}


def test_agent_keeps_app_prompt_and_tool_schemas_without_harness():
    model_func = agent_module.recruiting_agent.nodes["model"].bound.func
    closure = inspect.getclosurevars(model_func).nonlocals

    assert {tool.name for tool in closure["default_tools"]} == EXPECTED_TOOLS
    assert closure["system_message"].content == agent_module.SYSTEM_PROMPT
    assert set(agent_module.recruiting_agent.nodes) == {"__start__", "model", "tools"}


def test_run_agent_preserves_config_and_returns_final_message():
    calls = []

    class FakeAgent:
        def invoke(self, payload, config):
            calls.append((payload, config))
            return {"messages": [AIMessage(content="final reply")]}

    original_agent = agent_module.recruiting_agent
    agent_module.recruiting_agent = FakeAgent()
    try:
        result = agent_module.run_agent(
            "Find a candidate",
            user_id="REC-10001",
            environment="test",
            thread_id="thread-123",
        )
    finally:
        agent_module.recruiting_agent = original_agent

    assert result == "final reply"
    assert calls == [
        (
            {"messages": [{"role": "user", "content": "Find a candidate"}]},
            {
                "run_name": "Recruiting Assistant",
                "metadata": {
                    "thread_id": "thread-123",
                    "user_id": "REC-10001",
                    "environment": "test",
                },
            },
        )
    ]


def test_recruiter_lookup_is_required_as_first_step():
    assert "As a first step in any request, always call the get_current_recruiter tool" in agent_module.SYSTEM_PROMPT


def test_sent_email_uses_current_recruiter_signature():
    candidate = {"name": "Ada Lovelace", "email": "ada@example.com"}
    recruiter = agent_module.get_current_recruiter.invoke(
        {}, config={"metadata": {"user_id": "recruiter_amills"}}
    )

    result = agent_module.send_candidate_email.invoke(
        {
            "candidate": candidate,
            "subject": "Next steps",
            "body": "We would like to continue the conversation.",
            "from_recruiter": recruiter["recruiter"],
        }
    )

    assert result["status"] == "sent"
    assert result["from_name"] == recruiter["recruiter"]["name"]
    assert result["from"] == recruiter["recruiter"]["email"]
