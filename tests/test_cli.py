from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ometer.cli import build_parser, main, resolve_mode
from ometer.config import Config


class TestBuildParser:
    def _parse(self, *args: str):
        parser = build_parser()
        return parser.parse_args(args)

    def test_defaults(self):
        args = self._parse()
        assert args.local is False
        assert args.cloud is False
        assert args.model is None
        assert args.ttf is False
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

    def test_ttf_flag(self):
        args = self._parse("--ttf")
        assert args.ttf is True

    def test_tps_flag(self):
        args = self._parse("--tps")
        assert args.tps is True

    def test_verbose_flag(self):
        args = self._parse("--verbose")
        assert args.verbose is True

    def test_model_flag(self):
        args = self._parse("--model", "llama3")
        assert args.model == "llama3"


class TestResolveMode:
    def _args(self, **overrides):
        defaults = dict(
            local=False,
            cloud=False,
            model=None,
            ttf=False,
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
            self._args(model="llama3"), is_tty=True, prompt_fn=lambda: "both"
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
            patch("ometer.cli.stream_table", new_callable=AsyncMock) as mock_stream,
        ):
            await main("local", False, False, False, None, config)
            mock_stream.assert_called_once()
            assert mock_stream.call_args[0][3] == _LOCAL_MODELS

    @pytest.mark.asyncio
    async def test_local_fetch_error(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                side_effect=Exception("connection refused"),
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock),
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
            patch("ometer.cli.stream_table", new_callable=AsyncMock) as mock_stream,
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
            patch("ometer.cli.stream_table", new_callable=AsyncMock),
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
            patch("ometer.cli.stream_table", new_callable=AsyncMock),
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
            patch("ometer.cli.stream_table", new_callable=AsyncMock) as mock_stream,
        ):
            await main("cloud", False, True, False, None, config)
            assert mock_stream.call_args[0][8] == {"Authorization": "Bearer secret"}

    @pytest.mark.asyncio
    async def test_target_model_filters(self):
        models = [
            {"name": "llama3", "modified_at": "2024-01-01T00:00:00Z", "details": {}},
            {"name": "mistral", "modified_at": "2024-02-01T00:00:00Z", "details": {}},
        ]
        config = _make_config()
        with (
            patch("ometer.cli.fetch_tags", new_callable=AsyncMock, return_value=models),
            patch("ometer.cli.stream_table", new_callable=AsyncMock) as mock_stream,
        ):
            await main("local", False, False, False, "llama3", config)
            filtered = mock_stream.call_args[0][3]
            assert len(filtered) == 1
            assert filtered[0]["name"] == "llama3"

    @pytest.mark.asyncio
    async def test_target_model_not_found_exits(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_LOCAL_MODELS,
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock),
        ):
            with pytest.raises(SystemExit):
                await main("cloud", False, False, False, "nonexistent", config)

    @pytest.mark.asyncio
    async def test_target_model_suffix_in_title(self):
        config = _make_config()
        with (
            patch(
                "ometer.cli.fetch_tags",
                new_callable=AsyncMock,
                return_value=_LOCAL_MODELS,
            ),
            patch("ometer.cli.stream_table", new_callable=AsyncMock) as mock_stream,
        ):
            await main("local", False, False, False, "llama3", config)
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
            patch("ometer.cli.stream_table", new_callable=AsyncMock) as mock_stream,
        ):
            await main("cloud", False, False, False, "mistral", config)
            title = mock_stream.call_args[0][4]
            assert "mistral" in title

    @pytest.mark.asyncio
    async def test_no_local_models_message(self):
        config = _make_config()
        with (
            patch("ometer.cli.fetch_tags", new_callable=AsyncMock, return_value=[]),
            patch("ometer.cli.stream_table", new_callable=AsyncMock),
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
            patch("ometer.cli.stream_table", new_callable=AsyncMock),
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
            patch("ometer.cli.stream_table", new_callable=AsyncMock),
            patch("ometer.cli.console") as mock_console,
        ):
            await main(None, False, False, False, None, config)
            assert (
                any(c == () for c in mock_console.print.call_args_list)
                or mock_console.print.call_count >= 1
            )

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
            patch("ometer.cli.stream_table", new_callable=AsyncMock) as mock_stream,
        ):
            await main(None, False, False, False, None, config)
            assert mock_stream.call_count == 2


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
