from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
from collections import deque

from loguru import logger

from .config import TranslatorConfig
from .queue_utils import put_drop_oldest
from .types import EnSegment, ZhSegment

_TRANSLATION_TIMEOUT_S = 5.0

SYSTEM_PROMPT = (
    "你是专业的实时翻译员,把英文商务会议对话翻译成中文。\n"
    "规则:\n"
    "- 只输出中文翻译,严禁输出原文、解释、引号、Markdown\n"
    "- 口语化,贴近商务会议场景\n"
    "- 专有名词、产品名、人名保留英文(如 Salesforce、AWS、John)\n"
    "- 不确定的词宁可保留英文,不要瞎猜"
)


def alpha_count(s: str) -> int:
    return sum(1 for c in s if c.isascii() and c.isalpha())


def has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


def build_messages(
    system_prompt: str,
    history: list[tuple[str, str]],
    en_text: str,
) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for en, zh in history:
        msgs.append({"role": "user", "content": en})
        msgs.append({"role": "assistant", "content": zh})
    msgs.append({"role": "user", "content": en_text})
    return msgs


class TranslatorWorker(threading.Thread):
    def __init__(
        self,
        in_q: queue.Queue[EnSegment],
        out_q: queue.Queue[ZhSegment],
        stop_event: threading.Event,
        translator_config: TranslatorConfig,
        llm: object | None = None,
        persist_q: queue.Queue[ZhSegment] | None = None,
        pipeline_ref: object | None = None,
    ) -> None:
        super().__init__(name="TranslatorWorker", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.config = translator_config
        self.history: deque[tuple[str, str]] = deque(
            maxlen=translator_config.history_size
        )
        self._llm = llm  # if provided (tests), skip Llama load
        self.persist_q = persist_q
        self.pipeline_ref = pipeline_ref
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="LlamaCall"
        )

    def _load_llm(self) -> object:
        from llama_cpp import Llama

        logger.info(f"Loading Qwen translator: {self.config.model_path}")
        return Llama(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            flash_attn=True,
            verbose=False,
        )

    def _translate(self, en_text: str, temperature: float) -> str:
        messages = build_messages(SYSTEM_PROMPT, list(self.history), en_text)

        def _call() -> dict:
            return self._llm.create_chat_completion(  # type: ignore[union-attr]
                messages=messages,
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                stop=["\n\n"],
            )

        future = self._executor.submit(_call)
        try:
            resp = future.result(timeout=_TRANSLATION_TIMEOUT_S)
        except concurrent.futures.TimeoutError as e:
            # We can't actually cancel a running llama.cpp call, but we
            # can stop waiting for it. The next translation will queue
            # behind this one and may itself time out — acceptable for
            # the live use case (subtitle freshness > completeness).
            raise TimeoutError(
                f"translation exceeded {_TRANSLATION_TIMEOUT_S}s timeout"
            ) from e
        return resp["choices"][0]["message"]["content"].strip()

    def process_one(self, seg: EnSegment) -> ZhSegment | None:
        """Translate a single segment. Returns None when input is filtered out
        (too short) or the translation timed out."""
        if alpha_count(seg.text) < 2:
            logger.debug(f"Skip too-short EN: {seg.text!r}")
            return None

        t0 = time.perf_counter()
        try:
            zh = self._translate(seg.text, temperature=self.config.temperature)

            if not has_cjk(zh):
                logger.warning(
                    f"No CJK in translation, retrying temp=0.5: en={seg.text!r} zh={zh!r}"
                )
                zh = self._translate(seg.text, temperature=0.5)
                if not has_cjk(zh):
                    logger.warning(
                        f"Still no CJK after retry; emitting empty: en={seg.text!r}"
                    )
                    zh = ""
        except TimeoutError as e:
            logger.warning(f"Translation skipped: {e} (en={seg.text!r})")
            return None

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        self.history.append((seg.text, zh))

        return ZhSegment(
            en_text=seg.text,
            zh_text=zh,
            audio_ts_start=seg.audio_ts_start,
            audio_ts_end=seg.audio_ts_end,
            translation_latency_ms=elapsed_ms,
        )

    def run(self) -> None:
        backoff = 1.0
        attempts = 0
        while not self.stop_event.is_set():
            try:
                self._run_inner()
                return
            except Exception as e:
                attempts += 1
                logger.exception(
                    f"TranslatorWorker failed (attempt {attempts}/3): {e}"
                )
                if attempts >= 3:
                    logger.error("TranslatorWorker giving up; degraded mode")
                    return
                if self.stop_event.wait(backoff):
                    return
                backoff = min(backoff * 2, 4.0)

    def _run_inner(self) -> None:
        if self._llm is None:
            self._llm = self._load_llm()
        logger.info("Translator ready")

        while not self.stop_event.is_set():
            try:
                seg = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            zh_seg = self.process_one(seg)
            if zh_seg is None:
                continue
            logger.info(
                f"ZH [{zh_seg.translation_latency_ms}ms]: {zh_seg.zh_text}"
            )
            if self.pipeline_ref is not None:
                e2e_ms = int((time.time() - zh_seg.audio_ts_end) * 1000)
                self.pipeline_ref.last_e2e_ms = max(0, e2e_ms)  # type: ignore[attr-defined]
            put_drop_oldest(self.out_q, zh_seg, "subtitle_q")
            if self.persist_q is not None:
                put_drop_oldest(self.persist_q, zh_seg, "subtitle_q_persist")

        self._executor.shutdown(wait=False)
        logger.info("Translator stopped")
