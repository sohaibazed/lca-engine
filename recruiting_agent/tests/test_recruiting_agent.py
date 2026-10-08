import inspect
import json
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


class FakeScoringLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return agent_module.CandidateScore(
            score=80,
            justification="Good fit.",
            rubric_breakdown={
                "experience": 80,
                "skills_match": 80,
                "seniority_fit": 80,
            },
        )


def test_score_candidate_loads_full_profile_and_current_skills(monkeypatch):
    scorer = FakeScoringLLM()
    monkeypatch.setattr(agent_module, "_scoring_llm", scorer)

    result = agent_module.score_candidate.invoke(
        {"candidate_id": "CAND-71001", "job_id": "JOB-10005"}
    )

    assert result["score"] == 80
    assert len(scorer.calls) == 1
    user_message = scorer.calls[0][1]["content"]
    profile = json.loads(user_message.split("\n\nCandidate profile:\n", 1)[1])
    assert profile["work_history"]
    assert profile["years_experience"] == 6
    assert profile["skills"] == agent_module.data_service.fetch_skills("CAND-71001")


def test_score_candidate_rejects_unknown_candidate_without_llm(monkeypatch):
    scorer = FakeScoringLLM()
    monkeypatch.setattr(agent_module, "_scoring_llm", scorer)
    result = agent_module.score_candidate.invoke(
        {"candidate_id": "CAND-UNKNOWN", "job_id": "JOB-10005"}
    )

    assert result == {
        "score": None,
        "error": "Candidate CAND-UNKNOWN was not found.",
    }
    assert scorer.calls == []


def test_score_candidate_rejects_unknown_or_invalid_job_without_llm(monkeypatch):
    scorer = FakeScoringLLM()
    monkeypatch.setattr(agent_module, "_scoring_llm", scorer)
    unknown = agent_module.score_candidate.invoke(
        {"candidate_id": "CAND-71001", "job_id": "JOB-UNKNOWN"}
    )
    monkeypatch.setattr(
        agent_module.data_service,
        "get_job_posting",
        lambda job_id: {"required_skills": [], "min_years_experience": 1, "description": ""},
    )
    invalid = agent_module.score_candidate.invoke(
        {"candidate_id": "CAND-71001", "job_id": "JOB-INVALID"}
    )

    assert unknown["error"] == "Job JOB-UNKNOWN was not found."
    assert invalid["error"] == "Cannot score without a valid job description."
    assert scorer.calls == []


def test_build_candidate_profile_preserves_full_profile_result(monkeypatch):
    monkeypatch.setattr(
        agent_module.data_service,
        "CANDIDATES",
        {
            "CAND-TEST": {
                "name": "Test Candidate",
                "years_experience": 4,
                "work_history": [{"role": "Engineer"}],
                "education": [{"degree": "BS"}],
                "skills": ["Python"],
            }
        },
    )
    monkeypatch.setattr(
        agent_module.data_service,
        "get_profile_from_db",
        lambda candidate_id: {"candidate_profile": None},
    )
    monkeypatch.setattr(agent_module.data_service, "save_profile_to_db", lambda *args: None)

    result = agent_module.build_candidate_profile.invoke({"candidate_id": "CAND-TEST"})

    assert result == {
        "candidate_profile": {
            "candidate_id": "CAND-TEST",
            "name": "Test Candidate",
            "work_history": [{"role": "Engineer"}],
            "education": [{"degree": "BS"}],
            "skills": ["Python"],
            "years_experience": 4,
        },
        "found": True,
    }


def test_add_then_score_uses_persisted_skill_and_full_profile(monkeypatch):
    monkeypatch.setattr(
        agent_module.data_service,
        "CANDIDATES",
        {
            "CAND-TEST": {
                "name": "Test Candidate",
                "years_experience": 4,
                "work_history": [{"role": "Engineer"}],
                "education": [],
                "skills": ["Python"],
            }
        },
    )
    monkeypatch.setattr(
        agent_module.data_service,
        "get_job_posting",
        lambda job_id: {
            "required_skills": ["Python", "Go"],
            "min_years_experience": 2,
            "description": "Build services.",
        },
    )
    scorer = FakeScoringLLM()
    monkeypatch.setattr(agent_module, "_scoring_llm", scorer)
    events = []
    add_skill = agent_module.data_service.add_candidate_skill

    def tracked_add_skill(candidate_id, skill):
        result = add_skill(candidate_id, skill)
        events.append(("add", result["updated"]))
        return result

    monkeypatch.setattr(agent_module.data_service, "add_candidate_skill", tracked_add_skill)
    add_result = agent_module.add_candidate_skill.invoke(
        {"candidate_id": "CAND-TEST", "skill": "Go"}
    )
    score_result = agent_module.score_candidate.invoke(
        {"candidate_id": "CAND-TEST", "job_id": "JOB-TEST"}
    )
    events.append(("score", score_result["score"]))

    scored_profile = json.loads(
        scorer.calls[0][1]["content"].split("\n\nCandidate profile:\n", 1)[1]
    )
    assert add_result["updated"] is True
    assert agent_module.data_service.fetch_skills("CAND-TEST") == ["Python", "Go"]
    assert scored_profile["skills"] == ["Python", "Go"]
    assert scored_profile["work_history"] == [{"role": "Engineer"}]
    assert len(scorer.calls) == 1
    assert events == [("add", True), ("score", 80)]
    assert "call the next tool only after the previous tool succeeds" in agent_module.SYSTEM_PROMPT
