"""Regression tests: wiring failures must not vanish without a trace.

Three places where an exception was swallowed with a bare ``pass`` (or a
bare ``return {"error": ...}`` that dropped the traceback), so a genuine
failure looked like "the feature simply is not there":

* ``tools/__init__.py::_load_plugin_tools`` -- a plugin blowing up made
  EVERY plugin-provided tool disappear, and the agent then reported
  "Unknown tool 'X' requested, skipping..." with no hint of the cause;
* ``tools/command_runner.py`` -- the shell reader thread swallowed
  everything, so a reader that died mid-stream turned truncated output
  into a clean-looking result;
* ``tools/file_modifications.py`` -- two handlers returned the error
  string but dropped the traceback, unlike their neighbours in the same
  module.
"""

import logging
from unittest.mock import patch



class TestPluginToolRegistrationIsNotSilent:
    def test_plugin_failure_is_logged(self, caplog):
        """A plugin that explodes must leave a trace in the log."""
        from code_puppy.tools import _load_plugin_tools

        with (
            patch(
                "code_puppy.tools.on_register_tools",
                side_effect=RuntimeError("plugin blew up"),
            ),
            caplog.at_level(logging.ERROR),
        ):
            # Must not raise: a plugin failure cannot break core functionality.
            _load_plugin_tools()

        assert caplog.records, "the plugin failure must be logged"
        assert any("plugin" in record.getMessage().lower() for record in caplog.records)
        assert any(record.exc_info for record in caplog.records)


class TestShellReaderThreadIsNotSilent:
    def test_reader_failure_is_logged(self, caplog):
        """A reader thread that dies mid-stream must say so.

        Truncated output that looks like a clean result is the worst kind of
        silent failure, so the exception is logged even at debug level.
        """
        import subprocess
        import sys

        from code_puppy.tools import command_runner

        process = subprocess.Popen(
            [sys.executable, "-c", "print('hello')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            with caplog.at_level(logging.DEBUG):
                with patch.object(
                    command_runner.select, "select", side_effect=RuntimeError("boom")
                ):
                    command_runner.run_shell_command_streaming(
                        process, timeout=5, command="echo hello", silent=True
                    )
        finally:
            process.kill()
            process.wait(timeout=5)

        messages = [record.getMessage() for record in caplog.records]
        assert any("reader thread" in message.lower() for message in messages), (
            f"the reader failure must be logged, got: {messages}"
        )


class TestFileModificationErrorsKeepTheTraceback:
    def test_delete_snippet_logs_the_traceback(self, tmp_path):
        from code_puppy.tools import file_modifications

        target = tmp_path / "whatever.py"
        target.write_text("print('hi')\n")

        with patch.object(
            file_modifications,
            "_log_error",
        ) as log_error:
            with patch.object(
                file_modifications,
                "read_text_sanitized",
                side_effect=RuntimeError("read failed"),
            ):
                result = file_modifications._delete_snippet_from_file(
                    None, str(target), "print('hi')"
                )

        assert "error" in result
        log_error.assert_called_once()
        assert "delete_snippet_from_file" in log_error.call_args.args[0]

    def test_replace_in_file_logs_the_traceback(self, tmp_path):
        from code_puppy.tools import file_modifications

        target = tmp_path / "whatever.py"
        target.write_text("print('hi')\n")

        with patch.object(
            file_modifications,
            "_log_error",
        ) as log_error:
            with patch.object(
                file_modifications,
                "read_text_sanitized",
                side_effect=RuntimeError("read failed"),
            ):
                result = file_modifications._replace_in_file(
                    None, str(target), "print('hi')", "print('bye')"
                )

        assert "error" in result
        log_error.assert_called_once()
        assert "replace_in_file" in log_error.call_args.args[0]
