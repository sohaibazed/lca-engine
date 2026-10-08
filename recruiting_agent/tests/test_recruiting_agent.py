import inspect
import json
import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage
import pytest

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


@pytest.fixture
def candidate_state():
    candidate_id = "CAND-71001"
    record = agent_module.data_service.CANDIDATES[candidate_id]
    original_skills = list(record["skills"])
    cached_profile = agent_module.data_service._PROFILES.pop(candidate_id, None)
    try:
        yield candidate_id, record
    finally:
        record["skills"] = original_skills
        if cached_profile is not None:
            agent_module.data_service._PROFILES[candidate_id] = cached_profile
        else:
            agent_module.data_service._PROFILES.pop(candidate_id, None)


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


def test_add_candidate_skill_persists_skill(candidate_state):
    candidate_id, _ = candidate_state

    result = agent_module.data_service.add_candidate_skill(candidate_id, "distributed systems")

    assert result["updated"] is True
    assert "distributed systems" in agent_module.data_service.fetch_skills(candidate_id)


def test_add_candidate_skill_does_not_claim_success_when_write_fails(candidate_state, monkeypatch):
    candidate_id, record = candidate_state

    class FailingRecord(dict):
        def __setitem__(self, key, value):
            if key == "skills":
                raise OSError("write failed")
            super().__setitem__(key, value)

    monkeypatch.setitem(
        agent_module.data_service.CANDIDATES,
        candidate_id,
        FailingRecord(record),
    )

    with pytest.raises(OSError, match="write failed"):
        agent_module.data_service.add_candidate_skill(candidate_id, "distributed systems")


def test_build_candidate_profile_refreshes_after_skill_add(candidate_state):
    candidate_id, _ = candidate_state
    agent_module.build_candidate_profile.invoke({"candidate_id": candidate_id})

    agent_module.data_service.add_candidate_skill(candidate_id, "distributed systems")
    profile = agent_module.build_candidate_profile.invoke({"candidate_id": candidate_id})

    assert "distributed systems" in profile["candidate_profile"]["skills"]


def test_score_candidate_uses_persisted_added_skill(candidate_state, monkeypatch):
    candidate_id, _ = candidate_state
    agent_module.data_service.add_candidate_skill(candidate_id, "distributed systems")
    received_messages = []

    class FakeScoringLLM:
        def invoke(self, messages):
            received_messages.extend(messages)
            return SimpleNamespace(model_dump=lambda: {"score": 100})

    monkeypatch.setattr(agent_module, "_scoring_llm", FakeScoringLLM())
    agent_module.score_candidate.invoke(
        {
            "candidate_profile": {"candidate_id": candidate_id, "skills": []},
            "job_description": {
                "required_skills": ["distributed systems"],
                "min_years_experience": 1,
                "description": "Build distributed systems.",
            },
        }
    )

    scored_profile = json.loads(received_messages[1]["content"].split("\n\nCandidate profile:\n", 1)[1])
    assert "distributed systems" in scored_profile["skills"]
