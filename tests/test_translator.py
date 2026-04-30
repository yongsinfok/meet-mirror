from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock

from meet_mirror.config import TranslatorConfig
from meet_mirror.translator import (
    TranslatorWorker,
    alpha_count,
    build_messages,
    has_cjk,
)
from meet_mirror.types import EnSegment, ZhSegment


def test_alpha_count_counts_only_ascii_letters():
    assert alpha_count("hello") == 5
    assert alpha_count("hi 123") == 2
    assert alpha_count("中文") == 0
    assert alpha_count("a!") == 1
    assert alpha_count("") == 0


def test_has_cjk():
    assert has_cjk("中文")
    assert has_cjk("hello 世界")
    assert not has_cjk("hello")
    assert not has_cjk("")


def test_build_messages_with_history():
    msgs = build_messages(
        "SYS",
        [("hi", "你好"), ("bye", "再见")],
        "thanks",
    )
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "bye"},
        {"role": "assistant", "content": "再见"},
        {"role": "user", "content": "thanks"},
    ]


def _mk_worker(llm_mock: MagicMock, history_size: int = 5) -> TranslatorWorker:
    cfg = TranslatorConfig(
        model_path="dummy.gguf",
        n_gpu_layers=-1,
        n_ctx=8192,
        temperature=0.2,
        max_tokens=300,
        history_size=history_size,
    )
    return TranslatorWorker(
        in_q=queue.Queue(),
        out_q=queue.Queue(),
        stop_event=threading.Event(),
        translator_config=cfg,
        llm=llm_mock,
    )


def _mk_llm(responses: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": r}}]} for r in responses
    ]
    return llm


def test_short_input_skipped():
    llm = _mk_llm(["should not be called"])
    w = _mk_worker(llm)
    seg = EnSegment(text="a", audio_ts_start=0.0, audio_ts_end=1.0)
    assert w.process_one(seg) is None
    llm.create_chat_completion.assert_not_called()


def test_no_cjk_triggers_retry_at_higher_temperature():
    llm = _mk_llm(["english only output", "你好世界"])
    w = _mk_worker(llm)
    seg = EnSegment(text="hello world", audio_ts_start=1.0, audio_ts_end=2.0)
    out = w.process_one(seg)
    assert out is not None
    assert out.zh_text == "你好世界"
    assert llm.create_chat_completion.call_count == 2
    second_kwargs = llm.create_chat_completion.call_args_list[1].kwargs
    assert second_kwargs["temperature"] == 0.5


def test_no_cjk_after_retry_emits_empty():
    llm = _mk_llm(["english one", "english two"])
    w = _mk_worker(llm)
    seg = EnSegment(text="hello world", audio_ts_start=0.0, audio_ts_end=1.0)
    out = w.process_one(seg)
    assert out is not None
    assert out.zh_text == ""
    assert llm.create_chat_completion.call_count == 2


def test_history_keeps_last_n_pairs():
    llm = _mk_llm([f"翻译{i}" for i in range(10)])
    w = _mk_worker(llm, history_size=3)
    for i in range(7):
        seg = EnSegment(
            text=f"english {i}",
            audio_ts_start=float(i),
            audio_ts_end=float(i) + 1.0,
        )
        w.process_one(seg)
    assert len(w.history) == 3
    assert [en for en, _ in w.history] == ["english 4", "english 5", "english 6"]


def test_history_used_in_messages():
    llm = _mk_llm(["你好", "再见", "谢谢"])
    w = _mk_worker(llm, history_size=5)
    w.process_one(EnSegment(text="hello", audio_ts_start=0, audio_ts_end=1))
    w.process_one(EnSegment(text="goodbye", audio_ts_start=1, audio_ts_end=2))
    w.process_one(EnSegment(text="thanks", audio_ts_start=2, audio_ts_end=3))

    third_call_kwargs = llm.create_chat_completion.call_args_list[2].kwargs
    msgs = third_call_kwargs["messages"]
    # system + (hello,你好) + (goodbye,再见) + thanks
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "hello"}
    assert msgs[2] == {"role": "assistant", "content": "你好"}
    assert msgs[3] == {"role": "user", "content": "goodbye"}
    assert msgs[4] == {"role": "assistant", "content": "再见"}
    assert msgs[5] == {"role": "user", "content": "thanks"}


def test_zh_segment_fields_propagated():
    llm = _mk_llm(["翻译结果"])
    w = _mk_worker(llm)
    seg = EnSegment(text="hello world", audio_ts_start=10.0, audio_ts_end=15.5)
    out = w.process_one(seg)
    assert isinstance(out, ZhSegment)
    assert out.en_text == "hello world"
    assert out.zh_text == "翻译结果"
    assert out.audio_ts_start == 10.0
    assert out.audio_ts_end == 15.5
    assert out.translation_latency_ms >= 0
