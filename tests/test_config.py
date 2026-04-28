from __future__ import annotations

import os
from pathlib import Path

import pytest

from ometer.config import Config, _load_env


class TestLoadEnv:
    def test_loads_from_cwd_env(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("OLLAMA_LOCAL_BASE_URL=http://from-cwd:9999\n")
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent"))
        for key in (
            "OLLAMA_LOCAL_BASE_URL",
            "OLLAMA_CLOUD_BASE_URL",
            "OLLAMA_CLOUD_API_KEY",
            "OMETER_RUNS",
            "OMETER_PARALLEL",
        ):
            monkeypatch.delenv(key, raising=False)
        _load_env()
        assert os.getenv("OLLAMA_LOCAL_BASE_URL") == "http://from-cwd:9999"

    def test_loads_from_home_env(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        env_file = home / ".env"
        env_file.write_text("OLLAMA_CLOUD_API_KEY=mykey\n")
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(Path, "cwd", lambda: Path("/nonexistent-cwd"))
        for key in (
            "OLLAMA_LOCAL_BASE_URL",
            "OLLAMA_CLOUD_BASE_URL",
            "OLLAMA_CLOUD_API_KEY",
            "OMETER_RUNS",
            "OMETER_PARALLEL",
        ):
            monkeypatch.delenv(key, raising=False)
        _load_env()
        assert os.getenv("OLLAMA_CLOUD_API_KEY") == "mykey"

    def test_loads_from_config_dir(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        config_dir = home / ".config" / "ometer"
        config_dir.mkdir(parents=True)
        env_file = config_dir / ".env"
        env_file.write_text("OMETER_RUNS=1\n")
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(Path, "cwd", lambda: Path("/nonexistent-cwd"))
        for key in (
            "OLLAMA_LOCAL_BASE_URL",
            "OLLAMA_CLOUD_BASE_URL",
            "OLLAMA_CLOUD_API_KEY",
            "OMETER_RUNS",
            "OMETER_PARALLEL",
        ):
            monkeypatch.delenv(key, raising=False)
        _load_env()
        assert os.getenv("OMETER_RUNS") == "1"

    def test_skips_missing_paths(self, monkeypatch):
        monkeypatch.setattr(Path, "cwd", lambda: Path("/nonexistent-cwd-xyz"))
        monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent-home-xyz"))
        for key in (
            "OLLAMA_LOCAL_BASE_URL",
            "OLLAMA_CLOUD_BASE_URL",
            "OLLAMA_CLOUD_API_KEY",
            "OMETER_RUNS",
            "OMETER_PARALLEL",
        ):
            monkeypatch.delenv(key, raising=False)
        _load_env()

    def test_cwd_takes_priority(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        (cwd_dir / ".env").write_text("OLLAMA_LOCAL_BASE_URL=http://from-cwd\n")

        home_dir = tmp_path / "home"
        home_dir.mkdir()
        (home_dir / ".env").write_text("OLLAMA_LOCAL_BASE_URL=http://from-home\n")

        monkeypatch.setattr(Path, "cwd", lambda: cwd_dir)
        monkeypatch.setattr(Path, "home", lambda: home_dir)
        monkeypatch.delenv("OLLAMA_LOCAL_BASE_URL", raising=False)
        _load_env()
        assert os.getenv("OLLAMA_LOCAL_BASE_URL") == "http://from-cwd"


class TestConfigInit:
    def test_defaults(self):
        cfg = Config("http://localhost:11434", "https://ollama.com", "", 3, 1)
        assert cfg.local_base_url == "http://localhost:11434"
        assert cfg.cloud_base_url == "https://ollama.com"
        assert cfg.cloud_api_key == ""
        assert cfg.num_runs == 3
        assert cfg.num_parallel == 1

    def test_num_runs_clamped_low(self):
        cfg = Config("a", "b", "", 0, 1)
        assert cfg.num_runs == 1

    def test_num_runs_clamped_high(self):
        cfg = Config("a", "b", "", 10, 1)
        assert cfg.num_runs == 3

    def test_num_parallel_clamped_low(self):
        cfg = Config("a", "b", "", 1, 0)
        assert cfg.num_parallel == 1

    def test_num_parallel_clamped_high(self):
        cfg = Config("a", "b", "", 1, 20)
        assert cfg.num_parallel == 10

    def test_bench_prompts_active_matches_num_runs(self):
        cfg = Config("a", "b", "", 2, 1)
        assert len(cfg.bench_prompts_active) == 2

    def test_bench_prompts_active_single_run(self):
        cfg = Config("a", "b", "", 1, 1)
        assert len(cfg.bench_prompts_active) == 1


class TestConfigFromEnv:
    _ENV_KEYS = (
        "OLLAMA_LOCAL_BASE_URL",
        "OLLAMA_CLOUD_BASE_URL",
        "OLLAMA_CLOUD_API_KEY",
        "OMETER_RUNS",
        "OMETER_PARALLEL",
    )

    @pytest.fixture()
    def clean_env(self, monkeypatch):
        monkeypatch.setattr("ometer.config._load_env", lambda: None)
        for key in self._ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        yield

    def test_from_env_defaults(self, clean_env):
        cfg = Config.from_env()
        assert cfg.local_base_url == "http://localhost:11434"
        assert cfg.cloud_base_url == "https://ollama.com"
        assert cfg.cloud_api_key == ""
        assert cfg.num_runs == 3
        assert cfg.num_parallel == 1

    def test_from_env_custom_values(self, clean_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_LOCAL_BASE_URL", "http://host:1234")
        monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://cloud.example.com")
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "secret")
        monkeypatch.setenv("OMETER_RUNS", "2")
        monkeypatch.setenv("OMETER_PARALLEL", "5")
        cfg = Config.from_env()
        assert cfg.local_base_url == "http://host:1234"
        assert cfg.cloud_base_url == "https://cloud.example.com"
        assert cfg.cloud_api_key == "secret"
        assert cfg.num_runs == 2
        assert cfg.num_parallel == 5

    def test_from_env_invalid_runs_falls_back(self, clean_env, monkeypatch):
        monkeypatch.setenv("OMETER_RUNS", "abc")
        cfg = Config.from_env()
        assert cfg.num_runs == 3

    def test_from_env_invalid_parallel_falls_back(self, clean_env, monkeypatch):
        monkeypatch.setenv("OMETER_PARALLEL", "xyz")
        cfg = Config.from_env()
        assert cfg.num_parallel == 1

    def test_from_env_overrides_with_params(self, clean_env, monkeypatch):
        monkeypatch.setenv("OMETER_RUNS", "3")
        monkeypatch.setenv("OMETER_PARALLEL", "1")
        cfg = Config.from_env(runs=1, parallel=4)
        assert cfg.num_runs == 1
        assert cfg.num_parallel == 4
