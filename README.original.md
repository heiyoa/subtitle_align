# subtitle_align（Windows CPU-only）— 英文短剧字幕生成器（不漏字优先）

在 Windows 10/11（云电脑，CPU-only）生成英文短剧字幕，目标是“召回优先、不漏字”：宁可输出低置信度文本，也不要跳过有说话的片段。

- 输入：音频/视频 `mp4/mkv/mov/mp3/wav`
- 输出目录固定结构：
  - `transcript.json`：segments + word-level timestamps（来源：faster-whisper `word_timestamps`）
  - `subtitle.srt`
  - `subtitle.vtt`
  - `run_receipt.json`：参数、耗时、指标、降级/失败原因
  - `logs.txt`

## 安装（Windows cmd / PowerShell）

### 1) 安装 ffmpeg（必须）

确保命令行可用 `ffmpeg` / `ffprobe`：

```powershell
winget install Gyan.FFmpeg
ffmpeg -version
ffprobe -version
```

### 2) 安装 Python 依赖

```powershell
cd D:\zhuomian\手脚架\手脚架\Projects\subtitle_align
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 运行示例（3分钟视频）

```powershell
cd D:\zhuomian\手脚架\手脚架\Projects\subtitle_align

python -m subtitle_align `
  --input "D:\data\demo_3min.mp4" `
  --output "D:\zhuomian\手脚架\手脚架\Projects\subtitle_align\output\demo_3min" `
  --language auto `
  --model small `
  --compute_type int8 `
  --vad_filter off `
  --beam_size 5 `
  --enable_audio_enhance on `
  --hole_gap_sec 3.0 `
  --hole_pass_model small `
  --hole_beam_size 8 `
  --no_speech_threshold 0.3 `
  --log_prob_threshold -2.0 `
  --compression_ratio_threshold 3.0 `
  --threads 8
```

## 仅对现有 transcript 做“修复/过滤”（不跑 ASR）

当你的 Python 环境暂时无法安装/导入 `faster_whisper` 时，可以直接对已有输出目录里的 `transcript.json` 做后处理（应用 gates / 可选 OCR），快速修复“超长 working”等幻听字幕：

```powershell
cd D:\zhuomian\手脚架\手脚架\Projects\subtitle_align

python -m subtitle_align `
  --input "D:\zhuomian\手脚架\手脚架\参考剧集\Medical Genius Is Not Someone to Mess with-第1集.mp4" `
  --output "D:\zhuomian\手脚架\手脚架\参考剧集\第1集输出_gate" `
  --input_transcript "D:\zhuomian\手脚架\手脚架\参考剧集\第1集输出\transcript.json" `
  --enable_ocr off
```

## 抗幻听/非语音过滤（新增）

默认启用以下硬门控，目标：避免将呜咽/喘气/背景声识别成单词并生成超长字幕（例如把 `sob` 幻听成 `working` 且持续很久）。

- `--max_word_dur 1.2`：单词时长超过阈值直接剔除（可选 `--removed_word_placeholder ellipsis` 插入短 “…”）
- `--max_caption_dur_hard 6.0`：单条字幕超过硬上限会被强制重新切分，仍异常则丢弃并写入 warning
- `--island_window_sec 4.0` + `--island_max_words 2`：孤岛词（周围大空洞的 1-2 个词）剔除并记录 warning
- `--hallucination_lexicon PATH`：可选提供误识别词表（txt/json），默认内置包含 `working/okay/yeah/oh/...`

对应可观测性写入 `run_receipt.json`：
- `metrics.num_words_removed_by_gate`
- `metrics.removed_samples`（最多 10 条）
- `metrics.num_captions_resegmented`
- `warnings` 中会出现 `gate_*` 类信息

## 可选 OCR 校验（默认关闭，新增）

用途：仅当画面存在烧录字幕时，用 OCR 作为“校验信号”，不是主识别；OCR 失败不影响主流程。

参数：
- `--enable_ocr on|off`（默认 `off`）
- `--ocr_frames_per_caption 1|2`（默认 1）
- `--ocr_similarity_threshold 0.2`（默认 0.2）
- `--tesseract_cmd tesseract`（默认 `tesseract`）

安装 tesseract（Windows，免费）：
- 安装器：UB Mannheim Tesseract（安装后确保 `tesseract.exe` 在 PATH）
- 验证：`tesseract --version`

性能成本提示：
- OCR 会对每条字幕抽帧并执行一次（或两次）tesseract，整体耗时会显著增加；建议仅在“确实有烧录字幕”且需要校验时开启。

## 输出字段（摘要）

### `transcript.json`
- `raw_text_primary`：首次识别拼接文本
- `segments_primary[]`：首次识别 segments（每段含 `words[]`，每个 word 有 `word/start/end/probability`）
- `segments[]`：补洞合并后的 segments（用于最终字幕生成）
- `hole_filling.runs[]`：每个空洞区间的重跑信息（core/run 区间、模型、词数等）

### `run_receipt.json`
- `status`：`success | degraded | failed`
- `degraded_reason`：如 `no_word_timestamps` / `low_word_timestamp_coverage`
- `metrics.audio_duration_sec / processing_time_sec / rtf`
- `metrics.num_words / word_timestamp_coverage`
- `metrics.num_gaps_over_threshold`：补洞触发次数

运行 `python -m subtitle_align --help` 查看全部参数。
