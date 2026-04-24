"""Tests for the Ollama LLM agent (ollama_agent.py).

All OpenAI client calls are mocked — no Ollama server required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(content: str | None = None, tool_calls: list | None = None) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.model_dump.return_value = {"role": "assistant", "content": content}

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call(call_id: str, fn_name: str, args: dict) -> MagicMock:
    import json

    tc = MagicMock()
    tc.id = call_id
    tc.function.name = fn_name
    tc.function.arguments = json.dumps(args)
    return tc


# ---------------------------------------------------------------------------
# TOOLS schema
# ---------------------------------------------------------------------------


class TestToolsSchema:
    def test_all_tools_defined(self):
        from automl_model_training.ollama_agent import TOOLS

        names = {t["function"]["name"] for t in TOOLS}
        assert names == {
            "tool_profile",
            "tool_detect_leakage",
            "tool_engineer_features",
            "tool_train",
            "tool_predict",
            "tool_inspect_errors",
            "tool_read_analysis",
            "tool_compare_runs",
        }

    def test_tool_train_exposes_all_iteration_params(self):
        from automl_model_training.ollama_agent import TOOLS

        train = next(t for t in TOOLS if t["function"]["name"] == "tool_train")
        props = train["function"]["parameters"]["properties"]
        for param in [
            "preset",
            "problem_type",
            "eval_metric",
            "time_limit",
            "drop",
            "test_size",
            "seed",
            "prune",
            "cv_folds",
            "calibrate_threshold",
        ]:
            assert param in props, f"Missing parameter: {param}"

    def test_required_fields_present(self):
        from automl_model_training.ollama_agent import TOOLS

        for tool in TOOLS:
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_tool_map_matches_tools_list(self):
        from automl_model_training.ollama_agent import _TOOL_MAP, TOOLS

        schema_names = {t["function"]["name"] for t in TOOLS}
        assert schema_names == set(_TOOL_MAP.keys())


# ---------------------------------------------------------------------------
# run_ollama_agent — agent loop
# ---------------------------------------------------------------------------


class TestRunOllamaAgent:
    @patch("automl_model_training.ollama_agent.OpenAI")
    def test_stops_when_no_tool_calls(self, mock_openai_cls, tmp_path, capsys):
        from automl_model_training.ollama_agent import run_ollama_agent

        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="Training complete. Best model achieved F1=0.92."
        )

        run_ollama_agent(str(tmp_path / "data.csv"), "target", max_iterations=3)

        client.chat.completions.create.assert_called_once()
        captured = capsys.readouterr()
        assert "Training complete" in captured.out

    @patch("automl_model_training.ollama_agent._TOOL_MAP")
    @patch("automl_model_training.ollama_agent.OpenAI")
    def test_executes_tool_calls(self, mock_openai_cls, mock_tool_map, tmp_path):
        from automl_model_training.ollama_agent import run_ollama_agent

        client = MagicMock()
        mock_openai_cls.return_value = client

        tool_call = _make_tool_call(
            "call_1", "tool_profile", {"csv_path": "data.csv", "label": "target"}
        )
        mock_tool_map.__getitem__.return_value = MagicMock(return_value={"shape": [100, 3]})
        mock_tool_map.__contains__ = lambda self, key: True

        client.chat.completions.create.side_effect = [
            _make_response(tool_calls=[tool_call]),
            _make_response(content="Done."),
        ]

        run_ollama_agent(str(tmp_path / "data.csv"), "target", max_iterations=2)

        assert client.chat.completions.create.call_count == 2

    @patch("automl_model_training.ollama_agent._TOOL_MAP")
    @patch("automl_model_training.ollama_agent.OpenAI")
    def test_tool_error_returned_as_dict(self, mock_openai_cls, mock_tool_map, tmp_path):
        from automl_model_training.ollama_agent import run_ollama_agent

        client = MagicMock()
        mock_openai_cls.return_value = client

        tool_call = _make_tool_call("call_err", "tool_train", {"csv_path": "bad.csv", "label": "x"})
        mock_tool_map.__getitem__.return_value = MagicMock(side_effect=FileNotFoundError("bad.csv"))
        mock_tool_map.__contains__ = lambda self, key: True

        client.chat.completions.create.side_effect = [
            _make_response(tool_calls=[tool_call]),
            _make_response(content="Handled error."),
        ]

        # Should not raise — errors are caught and returned as {"error": ...}
        run_ollama_agent(str(tmp_path / "data.csv"), "target", max_iterations=2)
        assert client.chat.completions.create.call_count == 2

    @patch("automl_model_training.ollama_agent.OpenAI")
    def test_system_prompt_in_first_message(self, mock_openai_cls, tmp_path):
        from automl_model_training.ollama_agent import SYSTEM_PROMPT, run_ollama_agent

        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _make_response(content="Done.")

        run_ollama_agent(str(tmp_path / "data.csv"), "target")

        messages = client.chat.completions.create.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT

    @patch("automl_model_training.ollama_agent.OpenAI")
    def test_tools_and_tool_choice_passed_to_llm(self, mock_openai_cls, tmp_path):
        from automl_model_training.ollama_agent import TOOLS, run_ollama_agent

        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _make_response(content="Done.")

        run_ollama_agent(str(tmp_path / "data.csv"), "target")

        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["tools"] == TOOLS
        assert kwargs["tool_choice"] == "auto"

    @patch("automl_model_training.ollama_agent.OpenAI")
    def test_custom_model_and_base_url(self, mock_openai_cls, tmp_path):
        from automl_model_training.ollama_agent import run_ollama_agent

        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _make_response(content="Done.")

        run_ollama_agent(
            str(tmp_path / "data.csv"),
            "target",
            model="llama3.1:8b",
            base_url="http://custom:11434/v1",
        )

        mock_openai_cls.assert_called_once_with(base_url="http://custom:11434/v1", api_key="ollama")
        assert client.chat.completions.create.call_args[1]["model"] == "llama3.1:8b"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    @patch("automl_model_training.ollama_agent.run_ollama_agent")
    def test_cli_passes_args(self, mock_run, tmp_path):
        import sys

        from automl_model_training.ollama_agent import main

        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,0\n")

        sys.argv = [
            "agent-ollama",
            str(csv),
            "--label",
            "target",
            "--model",
            "llama3.1:8b",
            "--max-iterations",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
        main()

        mock_run.assert_called_once_with(
            csv_path=str(csv),
            label="target",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            max_iterations=3,
            output_dir=str(tmp_path),
        )
