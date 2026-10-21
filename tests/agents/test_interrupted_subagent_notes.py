"""Regression tests: interrupted sub-agent notes must not add user turns.

Background: the notes about interrupted sub-agents used to be appended to
the agent's message history as their own ``ModelRequest`` user messages.
The user's next message then became a *second* consecutive user turn, and
the Anthropic Messages API requires strict user/assistant alternation --
so the provider had to glue them together and the model ended up seeing a
conversation shape Code Puppy never built.

The notes now ride along inside the user's own turn.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from code_puppy.agents import _run_signals


class TestCollectInterruptedSubagentNotes:
    def test_returns_none_without_records(self):
        with patch(
            "code_puppy.tools.subagent_invocation.drain_interrupted_subagents",
            return_value=[],
        ):
            assert _run_signals.collect_interrupted_subagent_notes() is None

    def test_returns_the_note_text(self):
        records = [
            {
                "agent_name": "test-agent",
                "session_id": "sess-abc",
                "saved_count": 2,
            }
        ]
        with patch(
            "code_puppy.tools.subagent_invocation.drain_interrupted_subagents",
            return_value=records,
        ):
            note = _run_signals.collect_interrupted_subagent_notes()

        assert note is not None
        assert "test-agent" in note
        assert "sess-abc" in note
        assert "2 message(s)" in note

    def test_does_not_touch_the_message_history(self):
        """The whole point: no synthetic user message in the history.

        The collector no longer takes the agent at all -- it cannot touch
        the history, because it has no way to reach it.
        """
        records = [
            {"agent_name": "a", "session_id": "s1", "saved_count": 1},
            {"agent_name": "b", "session_id": "s2", "saved_count": None},
        ]
        with patch(
            "code_puppy.tools.subagent_invocation.drain_interrupted_subagents",
            return_value=records,
        ):
            note = _run_signals.collect_interrupted_subagent_notes()

        # Both notes travel in ONE block, so the user turn stays one turn.
        assert note is not None
        assert note.count("[system note]") == 2

    def test_saved_count_none_is_described(self):
        records = [{"agent_name": "a", "session_id": "s1", "saved_count": None}]
        with patch(
            "code_puppy.tools.subagent_invocation.drain_interrupted_subagents",
            return_value=records,
        ):
            note = _run_signals.collect_interrupted_subagent_notes()

        assert note is not None
        assert "no completed messages had been produced yet" in note


class TestRuntimePrependsTheNote:
    """The note must reach the model as part of the user's own turn."""

    def _drive(self, monkeypatch, note):
        """Run one turn and return the prompt the model actually received."""
        from types import SimpleNamespace

        from code_puppy.agents import _runtime, base_agent
        from code_puppy.agents.agent_code_puppy import CodePuppyAgent

        captured = {}

        agent = CodePuppyAgent()
        monkeypatch.setattr(agent, "get_model_name", lambda: "offline-model")
        monkeypatch.setattr(_runtime, "reset_pause_state_at_run_start", MagicMock())
        monkeypatch.setattr(
            _runtime, "collect_interrupted_subagent_notes", MagicMock(return_value=note)
        )
        monkeypatch.setattr(
            _runtime, "on_user_prompt_submit", AsyncMock(return_value=[])
        )
        monkeypatch.setattr(
            "code_puppy.model_switching.resolve_run_model_selection", MagicMock()
        )
        monkeypatch.setattr(_runtime, "_should_prepend_system_prompt", lambda _, p: p)

        def build(target, *, output_type=str, **kwargs):
            async def run(prompt, **options):
                captured["prompt"] = prompt
                return SimpleNamespace(data="ok", all_messages=lambda: [])

            target._code_generation_agent = SimpleNamespace(run=run)
            return target._code_generation_agent

        monkeypatch.setattr(_runtime, "build_pydantic_agent", build)
        monkeypatch.setattr(base_agent, "build_pydantic_agent", build)

        asyncio.run(_runtime._run_with_mcp_impl(agent, "hello there"))
        return captured.get("prompt")

    @staticmethod
    def _as_text(prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        parts = []
        for part in prompt:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)

    def test_note_is_prepended_to_the_prompt(self, monkeypatch):
        prompt = self._as_text(self._drive(monkeypatch, "[note]"))

        assert "[note]" in prompt
        assert "hello there" in prompt
        # The note comes FIRST: it is context for the user's question.
        assert prompt.index("[note]") < prompt.index("hello there")

    def test_nothing_is_prepended_without_notes(self, monkeypatch):
        prompt = self._as_text(self._drive(monkeypatch, None))

        assert "[note]" not in prompt
        assert "hello there" in prompt

    def test_the_user_turn_is_the_only_user_message(self, monkeypatch):
        """One user turn, so the provider never has to glue two together."""
        prompt = self._drive(monkeypatch, "[note]")

        if isinstance(prompt, str):
            return
        # Exactly one user part: the note rides inside it.
        assert len(prompt) == 1

    def test_the_note_survives_a_prompt_replacing_hook(self, monkeypatch):
        """A user_prompt_submit hook may REPLACE the prompt.

        The note must not be lost with it -- it is the only thing that tells
        the model the delegation was stopped.
        """
        from types import SimpleNamespace

        from code_puppy.agents import _runtime, base_agent
        from code_puppy.agents.agent_code_puppy import CodePuppyAgent

        captured = {}

        agent = CodePuppyAgent()
        monkeypatch.setattr(agent, "get_model_name", lambda: "offline-model")
        monkeypatch.setattr(_runtime, "reset_pause_state_at_run_start", MagicMock())
        monkeypatch.setattr(
            _runtime,
            "collect_interrupted_subagent_notes",
            MagicMock(return_value="[note]"),
        )
        # A hook that throws the user's prompt away.
        monkeypatch.setattr(
            _runtime,
            "on_user_prompt_submit",
            AsyncMock(return_value=["completely different prompt"]),
        )
        monkeypatch.setattr(
            "code_puppy.model_switching.resolve_run_model_selection", MagicMock()
        )
        monkeypatch.setattr(_runtime, "_should_prepend_system_prompt", lambda _, p: p)

        def build(target, *, output_type=str, **kwargs):
            async def run(prompt, **options):
                captured["prompt"] = prompt
                return SimpleNamespace(data="ok", all_messages=lambda: [])

            target._code_generation_agent = SimpleNamespace(run=run)
            return target._code_generation_agent

        monkeypatch.setattr(_runtime, "build_pydantic_agent", build)
        monkeypatch.setattr(base_agent, "build_pydantic_agent", build)

        asyncio.run(_runtime._run_with_mcp_impl(agent, "hello there"))

        prompt = self._as_text(captured.get("prompt"))
        assert "[note]" in prompt
        assert "completely different prompt" in prompt
