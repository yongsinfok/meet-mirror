# Meet Mirror — Phase 1 Implementation Plan

**Date:** 2026-04-30
**Status:** Ready to execute
**Spec:** `docs/superpowers/specs/2026-04-30-meet-mirror-design.md`
**Owner:** Joshua

---

## 总体策略

Phase 1 拆成 **7 个 vertical slice** (S0–S6),每个 slice 完成后系统都能独立跑起来,验证完再开下一个。每个 slice 一组 atomic commits,合并后立即可 dogfood。

**单 slice 节奏:**
1. 看本 slice 的 *Goal* 与 *Acceptance*
2. 按 *Tasks* 顺序写代码,逐 task 提交
3. 跑 *Verification* 步骤,失败回去 debug,不进下一个 slice
4. 最后 push,记录 *Notes* 到 commit message 或 BENCHMARKS.md

**Slice 之间不允许跳跃** —— 比如 Slice 3 的 Translator 必须建立在 Slice 2 ASR 已经稳定的基础上,否则两边 bug 会纠缠。

**模型权重不进 git** —— 由 `scripts/download_models.py` 拉取到 `models/`(已 gitignore)。

---

## Slice 0 — 项目骨架 (Skeleton)

**Goal:** `python main.py` 能跑起来,加载 config,打印 "Meet Mirror ready"。没有任何 ML、没有 UI。打底用。

### Tasks

| # | 文件 | 内容 |
|---|---|---|
| 0.1 | `pyproject.toml` | Python 3.11+,依赖:`numpy`, `pydantic`, `pyyaml`, `loguru`。开发依赖:`pytest`, `ruff`。`[project]` 名 `meet-mirror`,version `0.1.0`。 |
| 0.2 | `.gitignore` | `sessions/`、`models/`、`logs/`、`tests/fixtures/`、`__pycache__/`、`.venv/`、`*.egg-info/`、`.pytest_cache/`、`.ruff_cache/`、`config.local.yaml` |
| 0.3 | `src/meet_mirror/__init__.py` | 空文件 + `__version__ = "0.1.0"` |
| 0.4 | `src/meet_mirror/types.py` | `AudioChunk`、`EnSegment`、`ZhSegment` 三个 frozen dataclass(spec §5) |
| 0.5 | `src/meet_mirror/config.py` | Pydantic v2 `Config` 模型,字段对齐 spec §7。`Config.load(path)` 类方法读 yaml 并校验。 |
| 0.6 | `config.yaml` | spec §7 完整内容,paths 用相对路径 |
| 0.7 | `main.py` | 加载 config → loguru sink → `print("Meet Mirror ready")` → exit |
| 0.8 | `scripts/download_models.py` | 占位:打印需要下载的两个模型 URL(Whisper turbo via faster-whisper auto-download,Qwen2.5-7B-Instruct-Q4_K_M from huggingface),不实际下载 |
| 0.9 | `README.md` | 已存在,补充 *Quick start* 段:venv → `pip install -e .` → `python main.py` |

### Commits (建议)

- `chore: bootstrap python package skeleton`
- `feat(types): add AudioChunk/EnSegment/ZhSegment dataclasses`
- `feat(config): pydantic config loader with config.yaml`
- `chore(scripts): stub download_models.py`

### Verification

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
python main.py
# 预期输出:
# 2026-04-30 ... | INFO | Loaded config from config.yaml
# Meet Mirror ready
ruff check src/
pytest -q  # 无测试也应零退出
```

### Acceptance
- [ ] `python main.py` 0 退出
- [ ] `Config.load()` 校验失败时给出 Pydantic 错误信息
- [ ] `git status` 干净;`models/`、`sessions/` 不会被 add
- [ ] README *Quick start* 步骤跟着抄能跑通

---

## Slice 1 — 音频捕获 + 回放验证

**Goal:** WASAPI loopback 抓到 Teams/系统输出的 PCM 流,落到 wav 文件,人耳回放确认捕获正确。**没有 ASR/翻译。**

### Tasks

| # | 文件 | 内容 |
|---|---|---|
| 1.1 | `pyproject.toml` | 增加依赖:`soundcard`, `soundfile` |
| 1.2 | `src/meet_mirror/audio_capture.py` | `AudioCapture(threading.Thread)`:在 `run()` 里 `with sc.default_speaker().recorder(samplerate=16000, channels=1, blocksize=1600):` 循环 record(1600),拼装成 `AudioChunk` 推到 `self.out_q`。`stop_event` 控制退出。错误处理留 stub(Slice 6 再加 backoff)。 |
| 1.3 | `scripts/capture_test.py` | 起 `AudioCapture`,跑 10 秒,把所有 chunk 直接落到 `tests/fixtures/capture_test.wav`(int16 PCM,16 kHz)。退出后用 `soundfile.read()` + `sounddevice.play()` 回放。 |
| 1.4 | `tests/test_audio_capture.py` | 单测:mock `soundcard.default_speaker()`,推 3 个 fake block,断言 out_q 收到 3 个 AudioChunk,ts_start 单调递增,samples shape 正确 |

### Commits

- `feat(audio): WASAPI loopback capture worker`
- `chore(scripts): capture_test for manual loopback verification`
- `test(audio): unit test capture worker with mocked soundcard`

### Verification

```bash
# 在 Teams 里放一段音乐 / 找人通话
python scripts/capture_test.py
# 听 capture_test.wav 是否清晰、无卡顿、无削波
pytest tests/test_audio_capture.py -v
```

### Acceptance
- [ ] 10 秒回放清晰可懂,无明显爆音
- [ ] 关掉 Teams 把系统音频静音,再跑 → 全静音 wav,不报错
- [ ] `AudioChunk.ts_end - ts_start ≈ 0.1 s`(±5 ms)
- [ ] 单测过

### Notes
- 如果默认设备 sample rate 不是 16 kHz,`soundcard` 会自动 resample —— 在日志里确认实际 device samplerate,记录到 BENCHMARKS.md。
- Teams 音频需要"另一方说话"才有信号 —— 自己讲话不会进 loopback。

---

## Slice 2 — 加 ASR(英文转写到控制台)

**Goal:** 抓到的音频经 VAD 切句,Whisper turbo 转写,控制台实时打印英文 + 时间戳。**没有翻译/UI。**

### Tasks

| # | 文件 | 内容 |
|---|---|---|
| 2.1 | `pyproject.toml` | 增加依赖:`faster-whisper`, `torch`(CUDA 12.x 索引,README 写 `pip install --index-url https://download.pytorch.org/whl/cu124 ...`) |
| 2.2 | `src/meet_mirror/asr.py` | `AsrWorker(threading.Thread)`:加载 `WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")`。VAD 状态机(spec §6.2):accumulate voiced → 500 ms silence flush → 15 s 强制 flush → <300 ms 丢弃。每次 flush 调用 `model.transcribe(buf, language="en", vad_filter=False, beam_size=1, condition_on_previous_text=True)`,把 segments 拼成一条 `EnSegment` 推 `out_q`。 |
| 2.3 | `src/meet_mirror/pipeline.py` | `Pipeline`:拥有 `audio_q`(maxsize=200)、`asr_q`(maxsize=50)。`start()` 启动 `AudioCapture` 与 `AsrWorker`。`stop()` 设置 stop_event,join 5s。 |
| 2.4 | `main.py` | `--mode console` 模式:启动 Pipeline,起一个 consumer 线程从 `asr_q` 取 EnSegment 打印 `[ts] EN: text`。Ctrl+C 触发 `pipeline.stop()`。 |
| 2.5 | `tests/test_asr_vad.py` | 纯逻辑单测 VAD 状态机:用 fake voiced/silent samples 喂状态机,断言 flush 时机正确(在 silence_ms 后、在 max_utterance_s 后) |
| 2.6 | `scripts/benchmark.py` | v1:跑一段 30s 英文 wav,打印 Whisper transcribe wall time、tokens/s |

### Commits

- `feat(asr): faster-whisper worker with silero VAD state machine`
- `feat(pipeline): coordinator wiring audio→asr queues`
- `feat(main): console mode prints English transcription`
- `test(asr): VAD state machine unit tests`
- `chore(scripts): benchmark.py for whisper latency`

### Verification

```bash
python scripts/download_models.py  # 触发 faster-whisper 自动下载 turbo
python main.py --mode console
# 在 Teams 放 5–10 秒英文 → 看到:
# [00:00:03] EN: We need to align on the timeline before the demo.
# 延迟 ≤ 2 s
python scripts/benchmark.py tests/fixtures/sample_5s_en.wav
# 记录到 BENCHMARKS.md
pytest tests/ -q
```

### Acceptance
- [ ] 5 秒英文,从说完到打印延迟 < 2.0 s
- [ ] 长句(>15 s 不停)按 15 s 强制 flush,不丢音
- [ ] 静音 30 s 不卡死、不出空 segment
- [ ] VRAM 占用 < 2 GB(`nvidia-smi` 验证)
- [ ] benchmark 写入 BENCHMARKS.md

---

## Slice 3 — 加 Translator(中文打到控制台)

**Goal:** EnSegment → Qwen2.5-7B → 中文,控制台打印 EN/ZH 两行。**还没 UI。**

### Tasks

| # | 文件 | 内容 |
|---|---|---|
| 3.1 | `pyproject.toml` | 增加依赖:`llama-cpp-python`(README 写 CUDA wheel 安装命令:`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir`,或用预编译 cu124 wheel index) |
| 3.2 | `scripts/download_models.py` | 实装 Qwen 下载:从 HF `Qwen/Qwen2.5-7B-Instruct-GGUF` 下 `qwen2.5-7b-instruct-q4_k_m.gguf` 到 `models/`,SHA256 校验 |
| 3.3 | `src/meet_mirror/translator.py` | `TranslatorWorker(threading.Thread)`:加载 `Llama(model_path, n_gpu_layers=-1, n_ctx=8192, flash_attn=True, verbose=False)`。维护 `deque(maxlen=5)` 历史 (en, zh) 对作 few-shot。从 `in_q` 取 EnSegment → 跳过 <2 字母 → call `create_chat_completion`(system prompt 见 spec §6.3,temperature=0.2,max_tokens=300,stop=["\n\n"])→ CJK 检查(0 个汉字 → temp=0.5 重试 1 次)→ ZhSegment 推 `out_q`。 |
| 3.4 | `src/meet_mirror/pipeline.py` | 加 `subtitle_q`(maxsize=50),Pipeline 多启 `TranslatorWorker(asr_q → subtitle_q)`。 |
| 3.5 | `main.py` | console 模式 consumer 改成订阅 `subtitle_q`,打印 `[ts] EN: ... \n      ZH: ...` 两行,显示 `translation_latency_ms`。 |
| 3.6 | `tests/test_translator.py` | mock `Llama.create_chat_completion`,验证:(a) <2 字母输入跳过;(b) 无 CJK 触发重试;(c) 历史滑窗只保留 5 条;(d) 输出 ZhSegment 字段正确 |
| 3.7 | `scripts/benchmark.py` | 加 translator stage:输入 10 句典型商务英文,记录 p50/p95 latency |

### Commits

- `chore(scripts): download_models pulls Qwen2.5-7B-Q4 GGUF with sha256 check`
- `feat(translator): qwen2.5-7b llama.cpp worker with sliding history`
- `feat(translator): CJK validation + temperature retry`
- `feat(pipeline): wire translator into pipeline`
- `feat(main): console mode prints zh translation with latency`
- `test(translator): unit tests for filters and retry`

### Verification

```bash
python scripts/download_models.py  # 拉 Qwen GGUF (~4.4GB)
python main.py --mode console
# 真实 Teams 5 句英文,看到:
# [00:00:03] EN: We need to align on the timeline before the demo.
#            ZH: 我们需要在 demo 之前对齐时间线。 (lat=620ms)
nvidia-smi  # 双模型同载,~7.5–8.5 GB
python scripts/benchmark.py --translator
pytest tests/ -q
```

### Acceptance
- [ ] 端到端感知延迟 ≤ 2.5 s(spec §9 budget 是 3.5 s,留余量)
- [ ] 专有名词("Salesforce", 人名)保留英文
- [ ] 滑窗 5 条历史在多轮对话里维持代词一致性(主观验收 1 段会议录音)
- [ ] VRAM 总占用 ≤ 9 GB
- [ ] benchmark p95 ≤ 1500 ms

---

## Slice 4 — 浮动字幕 UI

**Goal:** 用 PyQt6 浮动字幕栏取代控制台输出。可拖动、淡入淡出、置顶、点击穿透。

### Tasks

| # | 文件 | 内容 |
|---|---|---|
| 4.1 | `pyproject.toml` | 增加依赖:`PyQt6` |
| 4.2 | `src/meet_mirror/subtitle_ui.py` | `SubtitleWindow(QWidget)`:flags `FramelessWindowHint | WindowStaysOnTopHint | Tool`,`WA_TranslucentBackground`,`WA_TransparentForMouseEvents`(可切)。`paintEvent` 画圆角半透明背景 + 白字 + 1px 黑描边。`QTimer(50ms)` 拉 `subtitle_q.get_nowait()`。淡入 `QPropertyAnimation` 200ms,hold 6s,淡出。双击切换 drag 模式(disable transparent-for-mouse + 加边框)。位置写回 `config.yaml`。 |
| 4.3 | `main.py` | 默认 mode 切回 GUI:`QApplication(sys.argv)` → `Pipeline()`(不 start)→ `SubtitleWindow(subtitle_q, config.subtitle)` → `pipeline.start()` → `app.exec()`。`--mode console` 保留作 debug。 |
| 4.4 | `src/meet_mirror/config.py` | `Config.save()` 方法,用于回写位置 |

### Commits

- `feat(ui): PyQt6 floating subtitle window with fade animation`
- `feat(ui): drag-to-reposition with double-click toggle`
- `feat(main): GUI mode is default; console mode behind flag`
- `feat(config): save() to persist subtitle position`

### Verification

```bash
python main.py
# - 字幕栏出现在屏幕底部居中
# - Teams 英文 → 5s 内中文出现并淡入
# - 6s 后淡出
# - 鼠标点击穿透到底层 Teams
# - 双击字幕 → 边框出现 → 拖到右上角 → 双击 → 锁定
# - 退出再启动 → 位置记住
```

### Acceptance
- [ ] 字幕栏不抢焦点(Teams 仍可输入)
- [ ] 多显示器:拖到副屏后位置正确持久化
- [ ] 字幕长度 > 一行时自动换行,最多 2 行,过长截断 + "…"
- [ ] 淡入淡出无闪烁;字体清晰(高 DPI 测试)

---

## Slice 5 — 持久化 + 托盘 + 全局热键

**Goal:** 一次会议产出 `sessions/<ts>/audio.wav` + `transcript.txt`。系统托盘可启停,全局快捷键 Ctrl+Alt+T。

### Tasks

| # | 文件 | 内容 |
|---|---|---|
| 5.1 | `pyproject.toml` | 增加:`pystray`, `Pillow`(pystray icon),`keyboard` |
| 5.2 | `src/meet_mirror/persistence.py` | `AudioWriter(threading.Thread)`:开 session 时 `soundfile.SoundFile(path, mode='w', samplerate=16000, channels=1, subtype='PCM_16')`,从 `audio_q_persist` 取 chunk → `.write(samples)`。stop 时 close。<br>`TextWriter(threading.Thread)`:从 `subtitle_q_persist` 取 ZhSegment → 追加 `[hh:mm:ss] EN: ... | ZH: ...\n`。 |
| 5.3 | `src/meet_mirror/pipeline.py` | 加两条 persist queue,AudioCapture/Translator 各推一份。Pipeline 启动时新建 `sessions/YYYY-MM-DD_HH-MM-SS/`,把路径传给 writer。`start()`/`stop()` 是幂等 toggle。 |
| 5.4 | `src/meet_mirror/tray.py` | `TrayApp`:`pystray.Icon` 跑后台线程。Icon 双色(绿=running, 灰=idle)。Menu:开始/停止、打开当前会话目录、重启、退出。`keyboard.add_hotkey("ctrl+alt+t", on_toggle)`,失败 fallback `pynput`。 |
| 5.5 | `main.py` | 新增 tray 启动 + hotkey 注册;退出时清理 tray + unhook。 |
| 5.6 | `tests/test_persistence.py` | AudioWriter 写 1 秒 fake audio,断言 wav 时长正确;TextWriter 喂 3 条 ZhSegment,断言 transcript.txt 行数与格式 |

### Commits

- `feat(persistence): audio_writer streams to wav, text_writer appends transcript`
- `feat(pipeline): session dir lifecycle + persist queues`
- `feat(tray): pystray menu + Ctrl+Alt+T global hotkey`
- `feat(main): wire tray + hotkey lifecycle`
- `test(persistence): writer unit tests`

### Verification

```bash
python main.py
# 托盘灰色 → Ctrl+Alt+T → 变绿 → 字幕开始工作
# 说英文 30 s
# Ctrl+Alt+T → 变灰
# 检查 sessions/2026-04-30_HH-MM-SS/audio.wav 时长 ≈ 30s,可播
# transcript.txt 每条带时间戳、EN/ZH 双语
# 托盘 → 打开当前会话目录 → explorer 打开
# 重启 → 字幕窗口位置/状态恢复
```

### Acceptance
- [ ] WAV header 完整(soundfile close 写入),用 VLC 能播
- [ ] transcript.txt 时间戳与 wav 时间对齐(±100 ms)
- [ ] Ctrl+Alt+T 跨多个 toggle 不泄漏 worker(线程数稳定)
- [ ] 退出程序后 sessions/ 文件可读、无锁

---

## Slice 6 — 加固:错误处理 + 日志 + Benchmarks + E2E

**Goal:** 把 spec §8 错误矩阵全部落地;长会议(60+ 分钟)不崩;有自动化 e2e 跑分。

### Tasks

| # | 模块 | 处理 |
|---|---|---|
| 6.1 | `audio_capture` | device disconnect → exponential backoff 1s/2s/4s,3 次失败后 tray 通知 |
| 6.2 | `asr` | CUDA OOM `RuntimeError` → 重新加载 `medium` model + tray 通知;后续不再升回 |
| 6.3 | `translator` | `create_chat_completion` 包 `concurrent.futures` 5s 超时,超时跳过 segment |
| 6.4 | `pipeline` | 每个 worker `run()` 顶层 try/except,异常 → 重启 ≤ 3 次,1s back-off,超出后 degraded 模式 + tray |
| 6.5 | 全部队列 | `put_nowait` `Full` → `get_nowait` 丢最旧 + log warning 含队列名 |
| 6.6 | `logging` | Loguru sink:控制台彩色 INFO + 文件 rotation 10MB DEBUG。每 30 s 打 status 行 `asr_q=N sub_q=N last_e2e_ms=NNN` |
| 6.7 | `scripts/e2e_test.py` | 输入 5 分钟英文会议 wav(`tests/fixtures/meeting_5min.wav`)→ pipeline 跑完 → 校验 sessions 输出存在、行数 > 阈值、p95 latency < 3.5 s,生成报告 |
| 6.8 | `BENCHMARKS.md` | 模板 + 首次正式测量结果(单机 RTX-X 16GB) |
| 6.9 | `tests/test_pipeline.py` | 集成测试(mock ASR/Translator):喂 fake AudioChunk → 断言 ZhSegment 落到 subtitle_q 与 persist_q,数量一致 |

### Commits

- `feat(audio): exponential backoff on device disconnect`
- `feat(asr): cuda oom fallback to medium model`
- `feat(translator): 5s timeout wrapper with skip-on-timeout`
- `feat(pipeline): worker restart with backoff and degraded mode`
- `feat(observability): loguru config + 30s status heartbeat`
- `chore(scripts): e2e_test.py with latency assertions`
- `docs: BENCHMARKS.md with first measurements`
- `test(pipeline): integration test with mocked ML workers`

### Verification

```bash
# 故障注入
# 1. 跑 main.py,中途拔耳机切设备 → 看到 backoff 日志,3s 内恢复
# 2. nvidia-smi 占满显存 → ASR fallback 到 medium,字幕继续
# 3. mock 翻译延迟 7s → 看到 timeout 跳过,无积压
python scripts/e2e_test.py tests/fixtures/meeting_5min.wav
# 报告显示 p95 < 3.5s
pytest tests/ -q
# 长跑测试
python main.py  # 接真实 Teams 会议 60+ 分钟
# 检查 logs/meet-mirror.log:无 ERROR,heartbeat 稳定
```

### Acceptance
- [ ] 所有 spec §8 错误场景都有对应代码路径(grep 验证)
- [ ] 60 分钟长跑无崩溃、无内存泄漏(峰值 RAM 不超过基线 + 500 MB)
- [ ] e2e_test.py CI-style 一键跑过
- [ ] BENCHMARKS.md 含 Whisper / Qwen / e2e 三档数据

---

## 跨 slice 通用规则

1. **每个 slice 的 commit 都用 atomic 风格**(`feat(scope):`, `fix(scope):`, `test(scope):`, `chore:`, `docs:`),便于 git log 追溯。
2. **不跳 verification** —— 即使时间紧,也要至少跑当 slice 的 *Acceptance* 第一条。
3. **失败回退优先于绕过** —— ASR OOM 就降模型,绝不删 condition_on_previous_text 这种核心特性来"省 VRAM"。
4. **真实数据 dogfood 优先** —— 每个 slice 完成后用一次真实 Teams 会议过一遍(哪怕 5 分钟),早发现感知问题。
5. **任何 slice 引入的新依赖必须同步更新 README** *Quick start* 段。
6. **`config.yaml` 改动必须保持向后兼容**(给新字段默认值),不要 break 已有 session。

---

## 不在本 plan 范围

- Phase 2(VibeVoice ASR + Markdown notes)— 单独 spec/plan
- 性能再压榨(LocalAgreement 流式 Whisper、Qwen Q3、speculative decoding)
- 多语种、TTS、麦克风路径
- 安装包打包(PyInstaller / MSIX)— Slice 6 完成稳定后再决定

---

## 时间估计(参考,非承诺)

| Slice | 难度 | 预估 |
|---|---|---|
| S0 骨架 | 低 | 0.5 day |
| S1 audio | 低 | 0.5 day |
| S2 ASR | 中 | 1.5 day(VAD 状态机调优是大头) |
| S3 translator | 中 | 1 day |
| S4 UI | 中 | 1.5 day(Qt 透明窗口在 Win11 有坑) |
| S5 persistence + tray | 中 | 1 day |
| S6 加固 | 中 | 1.5 day |
| **合计** | | **~7.5 工作日** |

---

## 启动 checklist

- [ ] Spec 已 commit、push
- [ ] `meet-mirror/` 当前只有 README + docs/(确认)
- [ ] CUDA 12.x toolkit 已装,`nvidia-smi` 看到显卡
- [ ] Python 3.11+ 在 PATH
- [ ] Teams 桌面客户端已装(用于 dogfood)
- [ ] 至少 30 GB 空闲磁盘(Whisper turbo + Qwen Q4 + sessions)

确认后从 **Slice 0 — Task 0.1** 开始。
