from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    capture_device: str = "default"
    sample_rate: int = 16000
    block_ms: int = 100


class AsrConfig(BaseModel):
    model: str = "large-v3-turbo"
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: Literal["float16", "int8", "int8_float16"] = "float16"
    language: str = "en"
    vad_threshold: float = 0.5
    silence_ms_to_flush: int = 500
    max_utterance_s: int = 15


class TranslatorConfig(BaseModel):
    model_path: str = "models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 8192
    temperature: float = 0.2
    max_tokens: int = 300
    history_size: int = 5


class SubtitlePosition(BaseModel):
    x: int | None = None
    y: int | None = None


class SubtitleConfig(BaseModel):
    font_family: str = "Microsoft YaHei"
    font_size: int = 24
    font_color: str = "#FFFFFF"
    background_color: str = "#000000"
    background_opacity: float = 0.7
    position: SubtitlePosition = Field(default_factory=SubtitlePosition)
    hold_seconds: int = 6


class SessionConfig(BaseModel):
    dir: str = "sessions"
    save_audio: bool = True
    save_transcript: bool = True


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "logs/meet-mirror.log"
    rotation_mb: int = 10


class Config(BaseModel):
    hotkey: str = "ctrl+alt+t"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    translator: TranslatorConfig = Field(default_factory=TranslatorConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                f,
                allow_unicode=True,
                sort_keys=False,
            )
