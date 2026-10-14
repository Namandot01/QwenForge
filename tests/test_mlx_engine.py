import platform

import pytest

from qwenforge.mlx_engine import EngineConfigurationError, NativeMLXEngine


def test_engine_refuses_non_macos_before_importing_mlx(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(EngineConfigurationError, match="macOS"):
        NativeMLXEngine("not-a-real-model")
