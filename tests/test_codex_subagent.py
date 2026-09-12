import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_demo.codex_subagent import ask_codex_subagent
from rag_demo.model_providers import ask_model, parse_model_spec


class CodexSubagentTests(unittest.TestCase):
    def test_provider_starts_ephemeral_tool_disabled_session(self):
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text("只根據證據回答。", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("rag_demo.codex_subagent.shutil.which", return_value="codex.exe"):
            with patch("rag_demo.codex_subagent.subprocess.run", side_effect=fake_run):
                answer = ask_codex_subagent(
                    "目前問題與證據",
                    model="gpt-5.5",
                    system="只能引用證據",
                )

        self.assertEqual(answer, "只根據證據回答。")
        command = captured["command"]
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--disable", command)
        self.assertIn("shell_tool", command)
        self.assertIn("memories", command)
        self.assertEqual(captured["kwargs"]["input"].count("目前問題與證據"), 1)
        self.assertIn("不得使用任何工具", captured["kwargs"]["input"])

    def test_provider_failure_is_reported(self):
        with patch("rag_demo.codex_subagent.shutil.which", return_value="codex.exe"):
            with patch(
                "rag_demo.codex_subagent.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="登入失敗",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Codex subagent failed"):
                    ask_codex_subagent("test")

    def test_model_provider_alias_and_dispatch(self):
        self.assertEqual(parse_model_spec("subagent:gpt-5.5").provider, "codex")
        with patch("rag_demo.codex_subagent.ask_codex_subagent", return_value="ok") as provider:
            self.assertEqual(ask_model("test", model="codex:gpt-5.5"), "ok")
        self.assertEqual(provider.call_args.kwargs["model"], "gpt-5.5")


if __name__ == "__main__":
    unittest.main()
