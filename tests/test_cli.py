from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ometer.cli import build_parser, main, main_entrypoint, match_model, resolve_mode
from ometer.config import Config
from ometer.export import ExportRow


class TestBuildParser:
    def _parse(self, *args: str):
        parser = build_parser()
        return parser.parse_args(args)

    def test_defaults(self):
        args = self._parse()
        assert args.local is False
        assert args.cloud is False
        assert args.model is None
        assert args.ttft is False
        assert args.tps is False
        assert args.verbose is False
        assert args.runs is None
        assert args.parallel is None

    def test_runs_flag(self):
        args = self._parse("--runs", "2")
        assert args.runs == 2

    def test_parallel_flag(self):
        args = self._parse("--parallel", "5")
        assert args.parallel == 5

    def test_num_predict_flag(self):
        args = self._parse("--num_predict", "128")
        assert args.num_predict == 128

    def test_runs_and_parallel(self):
        args = self._parse("--runs", "1", "--parallel", "3")
        assert args.runs == 1
        assert args.parallel == 3

    def test_invalid_runs_rejected(self):
        with pytest.raises(SystemExit):
            self._parse("--runs", "5")

    def test_local_flag(self):
        args = self._parse("--local")
        assert args.local is True
        assert args.cloud is False

    def test_cloud_flag(self):
        args = self._parse("--cloud")
        assert args.cloud is True
        assert args.local is False

    def test_local_and_cloud(self):
        args = self._parse("--local", "--cloud")
        assert args.local is True
        assert args.cloud is True

    def test_ttft_flag(self):
        args = self._parse("--ttft")
        assert args.ttft is True

    def test_tps_flag(self):
        args = self._parse("--tps")
        assert args.tps is True

    def test_verbose_flag(self):
        args = self._parse("--verbose")
        assert args.verbose is True

    def test_model_flag_single(self):
        args = self._parse("--model", "llama3")
        assert args.model == ["llama3"]

    def test_model_flag_multiple(self):
        args = self._parse("--model", "llama3", "mistral")
        assert args.model == ["llama3", "mistral"]

    def test_sort_flag(self):
        args = self._parse("--sort", "name")
        assert args.sort == "name"
        assert args.reverse is False

    def test_reverse_flag(self):
        args = self._parse("--reverse")
        assert args.reverse is True
        assert args.sort is None

    def test_sort_and_reverse_combined(self):
        args = self._parse("--sort", "ttft", "--reverse")
        assert args.sort == "ttft"
        assert args.reverse is True

    def test_reverse_without_sort_rejected(self):
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["ometer", "--reverse"]
            with pytest.raises(SystemExit):
                main_entrypoint()
        finally:
            sys.argv = old_argv


class TestResolveMode:
    def _args(self, **overrides):
        defaults = dict(
            local=False,
            cloud=False,
            model=None,
            ttft=False,
            tps=False,
            verbose=False,
            runs=None,
            parallel=None,
        )
        defaults.update(overrides)
        import argparse

        return argparse.Namespace(**defaults)

    def test_local_only(self):
        mode = resolve_mode(
            self._args(local=True), is_tty=True, prompt_fn=lambda: "both"
        )
        assert mode == "local"

    def test_cloud_only(self):
        mode = resolve_mode(
            self._args(cloud=True), is_tty=True, prompt_fn=lambda: "both"
        )
        assert mode == "cloud"

    def test_both_flags_returns_none(self):
        mode = resolve_mode(
            self._args(local=True, cloud=True), is_tty=True, prompt_fn=lambda: "both"
        )
        assert mode is None

    def test_no_flags_non_tty(self):
        mode = resolve_mode(self._args(), is_tty=False, prompt_fn=lambda: "both")
        assert mode is None

    def test_no_flags_with_model_set(self):
        mode = resolve_mode(
            self._args(model=["llama3"]), is_tty=True, prompt_fn=lambda: "both"
        )
        assert mode is None

    def test_prompt_returns_local(self):
        mode = resolve_mode(self._args(), is_tty=True, prompt_fn=lambda: "local")
        assert mode == "local"

    def test_prompt_returns_cloud(self):
        mode = resolve_mode(self._args(), is_tty=True, prompt_fn=lambda: "cloud")
        assert mode == "cloud"

    def test_prompt_returns_both(self):
        mode = resolve_mode(self._args(), is_tty=True, prompt_fn=lambda: "both")
        assert mode is None

    def test_prompt_returns_cancel(self):
        with pytest.raises(SystemExit):
            resolve_mode(self._args(), is_tty=True, prompt_fn=lambda: "cancel")


class TestMatchModel:
    def test_exact_match(self):
        assert match_model("llama3:latest", "llama3:latest") is True

    def test_family_match_tagged_model(self):
        assert match_model("llama3:latest", "llama3") is True

    def test_family_match_tagged_query(self):
        assert match_model("llama3", "llama3:latest") is True

    def test_family_match_both_tagged(self):
        assert match_model("llama3:8b", "llama3") is True

    def test_exact_match_with_tag(self):
        assert match_model("llama3:8b", "llama3:8b") is True

    def test_different_family(self):
        assert match_model("mistral:latest", "llama3") is False

    def test_no_partial_match(self):
        assert match_model("codellama:7b", "code") is False

    def test_empty_target_returns_false(self):
        assert match_model("llama3:latest", "") is False

    def test_no_false_partial(self):
        assert match_model("phi3:latest", "phi2") is False


class TestVersion:
    def test_version_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0


class TestExportFlags:
    def test_json_flag_no_path(self):
        parser = build_parser()
        args = parser.parse_args(["--json"])
        assert args.json_export == ""

    def test_json_flag_with_path(self):
        parser = build_parser()
        args = parser.parse_args(["--json", "/tmp/out.json"])
        assert args.json_export == "/tmp/out.json"

    def test_csv_flag_no_path(self):
        parser = build_parser()
        args = parser.parse_args(["--csv"])
        assert args.csv_export == ""

    def test_csv_flag_with_path(self):
        parser = build_parser()
        args = parser.parse_args(["--csv", "/tmp/out.csv"])
        assert args.csv_export == "/tmp/out.csv"

    def test_json_and_csv_mutually_exclusive(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--json", "--csv"])

    def test_no_export_flags(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.json_export is None
        assert args.csv_export is None


_LOCAL_MODELS = [
    {
        "name": "llama3",
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {"parameter_size": "8B"},
    },
]
_CLOUD_MODELS = [
    {
        "name": "mistral",
        "modified_at": "2024-02-01T00:00:00Z",
        "details": {"parameter_size": "7B"},
    },
]


def _make_config(**kwargs: str | int):
    return Config(
        local_base_url=str(kwargs.get("local_base_url", "http://localhost:11434")),
        cloud_base_url=str(kwargs.get("cloud_base_url", "https://ollama.com")),
        cloud_api_key=str(kwargs.get("cloud_api_key", "")),
        num_runs=int(kwargs.get("num_runs", 1)),
        num_parallel=int(kwargs.get("num_parallel", 1)),
    )


class TestMain:
    @pytest.mark.asyncio
    async def test_local_only_fetches_local_models(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_LOCAL_MODELS,
            ),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("local", False, False, False, None, config)
            mock_stream.assert_called_once()
            assert mock_stream.call_args[0][3] == _LOCAL_MODELS

    @pytest.mark.asyncio
    async def test_main_passes_num_predict(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_LOCAL_MODELS,
            ),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("local", False, False, False, None, config, 64)
            assert mock_stream.call_args.kwargs["num_predict"] == 64

    @pytest.mark.asyncio
    async def test_local_fetch_error(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                side_effect=Exception("connection refused"),
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.console") as mock_console,
        ):
            await main("local", False, False, False, None, config)
            mock_console.print.assert_called()
            assert any(
                "Skipping local" in str(c) for c in mock_console.print.call_args_list
            )

    @pytest.mark.asyncio
    async def test_cloud_only_fetches_cloud_models(self):
        config = _make_config(cloud_api_key="testkey")
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_CLOUD_MODELS,
            ),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("cloud", False, False, False, None, config)
            mock_stream.assert_called_once()
            assert mock_stream.call_args[0][3] == _CLOUD_MODELS

    @pytest.mark.asyncio
    async def test_cloud_fetch_error(self):
        config = _make_config(cloud_api_key="testkey")
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                side_effect=Exception("timeout"),
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.console") as mock_console,
        ):
            await main("cloud", False, False, False, None, config)
            mock_console.print.assert_called()
            assert any(
                "Failed to fetch cloud" in str(c)
                for c in mock_console.print.call_args_list
            )

    @pytest.mark.asyncio
    async def test_cloud_benchmark_warning_no_key(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                side_effect=Exception("skip"),
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.console") as mock_console,
        ):
            await main("cloud", True, False, False, None, config)
            assert any(
                "OLLAMA_CLOUD_API_KEY" in str(c)
                for c in mock_console.print.call_args_list
            )

    @pytest.mark.asyncio
    async def test_cloud_headers_with_api_key(self):
        config = _make_config(cloud_api_key="secret")
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_CLOUD_MODELS,
            ),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("cloud", False, True, False, None, config)
            assert mock_stream.call_args[0][8] == {"Authorization": "Bearer secret"}

    @pytest.mark.asyncio
    async def test_target_model_filters_exact(self):
        models = [
            {"name": "llama3", "modified_at": "2024-01-01T00:00:00Z", "details": {}},
            {"name": "mistral", "modified_at": "2024-02-01T00:00:00Z", "details": {}},
        ]
        config = _make_config()
        with (
            patch("ometer.cli.fetch_tags", new_callable=AsyncMock, return_value=models),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("local", False, False, False, ["llama3"], config)
            filtered = mock_stream.call_args[0][3]
            assert len(filtered) == 1
            assert filtered[0]["name"] == "llama3"

    @pytest.mark.asyncio
    async def test_target_model_family_match(self):
        models = [
            {
                "name": "llama3:latest",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {},
            },
            {
                "name": "mistral:7b",
                "modified_at": "2024-02-01T00:00:00Z",
                "details": {},
            },
        ]
        config = _make_config()
        with (
            patch("ometer.cli.fetch_tags", new_callable=AsyncMock, return_value=models),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("local", False, False, False, ["llama3"], config)
            filtered = mock_stream.call_args[0][3]
            assert len(filtered) == 1
            assert filtered[0]["name"] == "llama3:latest"

    @pytest.mark.asyncio
    async def test_target_model_multiple(self):
        models = [
            {"name": "llama3", "modified_at": "2024-01-01T00:00:00Z", "details": {}},
            {"name": "mistral", "modified_at": "2024-02-01T00:00:00Z", "details": {}},
            {"name": "phi3", "modified_at": "2024-03-01T00:00:00Z", "details": {}},
        ]
        config = _make_config()
        with (
            patch("ometer.cli.fetch_tags", new_callable=AsyncMock, return_value=models),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("local", False, False, False, ["llama3", "phi3"], config)
            filtered = mock_stream.call_args[0][3]
            assert len(filtered) == 2
            names = [m["name"] for m in filtered]
            assert "llama3" in names
            assert "phi3" in names

    @pytest.mark.asyncio
    async def test_target_model_not_found_exits(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_LOCAL_MODELS,
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]),
        ):
            with pytest.raises(SystemExit):
                await main("cloud", False, False, False, ["nonexistent"], config)

    @pytest.mark.asyncio
    async def test_target_model_suffix_in_title(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_LOCAL_MODELS,
            ),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("local", False, False, False, ["llama3"], config)
            title = mock_stream.call_args[0][4]
            assert "llama3" in title

    @pytest.mark.asyncio
    async def test_target_model_suffix_in_cloud_title(self):
        config = _make_config(cloud_api_key="key")
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_CLOUD_MODELS,
            ),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main("cloud", False, False, False, ["mistral"], config)
            title = mock_stream.call_args[0][4]
            assert "mistral" in title

    @pytest.mark.asyncio
    async def test_no_local_models_message(self):
        config = _make_config()
        with (
            patch("ometer.cli.fetch_tags", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.console") as mock_console,
        ):
            await main("local", False, False, False, None, config)
            assert any(
                "No local models" in str(c) for c in mock_console.print.call_args_list
            )

    @pytest.mark.asyncio
    async def test_no_cloud_models_message(self):
        config = _make_config(cloud_api_key="key")
        with (
            patch("ometer.cli.fetch_tags", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.console") as mock_console,
        ):
            await main("cloud", False, False, False, None, config)
            assert any(
                "No cloud models" in str(c) for c in mock_console.print.call_args_list
            )

    @pytest.mark.asyncio
    async def test_both_modes_prints_separator(self):
        config = _make_config()
        local_models = [
            {"name": "llama3", "modified_at": "2024-01-01T00:00:00Z", "details": {}}
        ]
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                side_effect=[local_models, []],
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.console") as mock_console,
        ):
            await main(None, False, False, False, None, config)
            mock_console.print.assert_any_call()

    @pytest.mark.asyncio
    async def test_both_local_and_cloud_models(self):
        config = _make_config(cloud_api_key="key")
        local_models = [
            {"name": "llama3", "modified_at": "2024-01-01T00:00:00Z", "details": {}}
        ]
        cloud_models = [
            {"name": "mistral", "modified_at": "2024-02-01T00:00:00Z", "details": {}}
        ]
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                side_effect=[local_models, cloud_models],
            ),
            patch(
                "ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]
            ) as mock_stream,
        ):
            await main(None, False, False, False, None, config)
            assert mock_stream.call_count == 2

    @pytest.mark.asyncio
    async def test_export_json_to_stdout(self):
        config = _make_config()
        export_row = ExportRow(
            model="llama3",
            size="8B",
            context="4096",
            quant="Q4_0",
            capabilities="completion",
            ttft=1.0,
            tps=50.0,
            error=None,
            runs=[{"prompt": "hi", "ttft": 1.0, "tps": 50.0, "error": None}],
        )
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_LOCAL_MODELS,
            ),
            patch(
                "ometer.cli.stream_table",
                new_callable=AsyncMock,
                return_value=[export_row],
            ),
            patch("ometer.cli.export_results") as mock_export,
        ):
            await main(
                "local",
                True,
                True,
                False,
                None,
                config,
                export_fmt="json",
                export_path=None,
            )
            mock_export.assert_called_once()
            call_args = mock_export.call_args
            assert call_args[0][0] == [export_row]
            assert call_args[0][1] == "json"
            assert call_args[0][2] is None

    @pytest.mark.asyncio
    async def test_export_csv_to_file(self, tmp_path):
        config = _make_config()
        export_row = ExportRow(
            model="llama3",
            size="8B",
            context="4096",
            quant="Q4_0",
            capabilities="completion",
            ttft=1.0,
            tps=50.0,
            error=None,
            runs=[{"prompt": "hi", "ttft": 1.0, "tps": 50.0, "error": None}],
        )
        path = str(tmp_path / "out.csv")
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_LOCAL_MODELS,
            ),
            patch(
                "ometer.cli.stream_table",
                new_callable=AsyncMock,
                return_value=[export_row],
            ),
            patch("ometer.cli.export_results") as mock_export,
        ):
            await main(
                "local",
                True,
                True,
                False,
                None,
                config,
                export_fmt="csv",
                export_path=path,
            )
            mock_export.assert_called_once()
            assert mock_export.call_args[0][1] == "csv"
            assert mock_export.call_args[0][2] == path

    @pytest.mark.asyncio
    async def test_export_skipped_when_no_models(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                side_effect=Exception("skip"),
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.console"),
            patch("ometer.cli.export_results") as mock_export,
        ):
            await main(
                "local",
                True,
                True,
                False,
                None,
                config,
                export_fmt="json",
                export_path=None,
            )
            mock_export.assert_not_called()


def _close_coro(coro):
    coro.close()


def _close_coro_then_raise_ki(coro):
    coro.close()
    raise KeyboardInterrupt()


class TestMainEntrypoint:
    def test_normal_flow(self):
        config = _make_config()
        with (
            patch("ometer.cli.Config") as mock_config_cls,
            patch("ometer.cli.resolve_mode", return_value=None),
            patch("ometer.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("sys.argv", ["ometer", "--local"]),
        ):
            mock_config_cls.from_env.return_value = config
            from ometer.cli import main_entrypoint

            main_entrypoint()
            mock_run.assert_called_once()

    def test_cancel_prompt(self):
        config = _make_config()
        with (
            patch("ometer.cli.Config") as mock_config_cls,
            patch("ometer.cli.resolve_mode", side_effect=SystemExit(0)),
            patch("ometer.cli.console") as mock_console,
            patch("sys.argv", ["ometer"]),
        ):
            mock_config_cls.from_env.return_value = config
            from ometer.cli import main_entrypoint

            with pytest.raises(SystemExit):
                main_entrypoint()
            assert any("Canceled" in str(c) for c in mock_console.print.call_args_list)

    def test_keyboard_interrupt(self):
        config = _make_config()
        with (
            patch("ometer.cli.Config") as mock_config_cls,
            patch("ometer.cli.resolve_mode", return_value="local"),
            patch("ometer.cli.asyncio.run", side_effect=_close_coro_then_raise_ki),
            patch("ometer.cli.console"),
            patch("sys.argv", ["ometer", "--local"]),
        ):
            mock_config_cls.from_env.return_value = config
            from ometer.cli import main_entrypoint

            with pytest.raises(SystemExit) as exc_info:
                main_entrypoint()
            assert exc_info.value.code == 130

    def test_interactive_prompt_called(self):
        config = _make_config()
        with (
            patch("ometer.cli.Config") as mock_config_cls,
            patch("ometer.cli.ListPrompt") as mock_list_prompt,
            patch("ometer.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("sys.argv", ["ometer"]),
        ):
            mock_config_cls.from_env.return_value = config
            mock_list_prompt.return_value.execute.return_value = "both"
            from ometer.cli import main_entrypoint

            with patch("ometer.cli.sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                main_entrypoint()
            mock_list_prompt.assert_called_once()
            mock_run.assert_called_once()

    def test_json_flag_resolution(self):
        config = _make_config()
        captured = {}

        def _run_and_capture(coro):
            captured["args"] = coro.cr_frame.f_locals
            coro.close()

        with (
            patch("ometer.cli.Config") as mock_config_cls,
            patch("ometer.cli.resolve_mode", return_value="local"),
            patch("ometer.cli.asyncio.run", side_effect=_run_and_capture),
            patch("sys.argv", ["ometer", "--local", "--json"]),
        ):
            mock_config_cls.from_env.return_value = config
            from ometer.cli import main_entrypoint

            main_entrypoint()
            assert captured["args"]["export_fmt"] == "json"
            assert captured["args"]["export_path"] is None

    def test_json_flag_with_path_resolution(self):
        config = _make_config()
        captured = {}

        def _run_and_capture(coro):
            captured["args"] = coro.cr_frame.f_locals
            coro.close()

        with (
            patch("ometer.cli.Config") as mock_config_cls,
            patch("ometer.cli.resolve_mode", return_value="local"),
            patch("ometer.cli.asyncio.run", side_effect=_run_and_capture),
            patch("sys.argv", ["ometer", "--local", "--json", "/tmp/out.json"]),
        ):
            mock_config_cls.from_env.return_value = config
            from ometer.cli import main_entrypoint

            main_entrypoint()
            assert captured["args"]["export_fmt"] == "json"
            assert captured["args"]["export_path"] == "/tmp/out.json"

    def test_csv_flag_resolution(self):
        config = _make_config()
        captured = {}

        def _run_and_capture(coro):
            captured["args"] = coro.cr_frame.f_locals
            coro.close()

        with (
            patch("ometer.cli.Config") as mock_config_cls,
            patch("ometer.cli.resolve_mode", return_value="local"),
            patch("ometer.cli.asyncio.run", side_effect=_run_and_capture),
            patch("sys.argv", ["ometer", "--local", "--csv"]),
        ):
            mock_config_cls.from_env.return_value = config
            from ometer.cli import main_entrypoint

            main_entrypoint()
            assert captured["args"]["export_fmt"] == "csv"
            assert captured["args"]["export_path"] is None
