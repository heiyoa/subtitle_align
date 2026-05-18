# 精准对齐字幕工具

英文目录名：`subtitle_align`

## 概述

`subtitle_align` 是一个独立的字幕生成与时间对齐模块，用于在音视频处理链路中提供稳定的基础字幕能力。重点在于中间结果可复查、运行状态可回执。

## 核心模块

- `subtitle_align/`：核心字幕生成与对齐逻辑
- `tests/`：基础规则测试
- `CODE_INDEX.md`：模块定位索引

## 架构思路

```text
Media Input
    │
    ▼
 ASR / Transcript Builder
    │   生成转写与词级时间信息
    ▼
 Alignment Layer
    │   处理时间对齐与后处理
    ▼
 Output Layer
    ├── subtitle.srt
    ├── subtitle.vtt
    ├── transcript.json
    └── run_receipt.json
```

## 工作流说明

```text
1. 输入音频或视频
2. 生成转写结果
3. 对齐时间戳
4. 输出字幕文件
5. 保留中间结果与运行回执
```

## 容错设计

```text
- transcript.json 保留：便于补处理和人工检查
- run_receipt.json 独立输出：便于判断是否降级或异常
- OCR 校验为可选项：附加能力异常不影响主流程
- 字幕层独立：避免问题扩散到更复杂的视频流程
```

## 当前状态

当前副本保留完整代码、测试和说明文档，可作为视频工作流基础模块展示。
