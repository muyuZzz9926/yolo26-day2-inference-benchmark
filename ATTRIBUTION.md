# 来源与许可声明 / Attribution

## 上游课程仓库

本仓库是基于下面这份公开课程材料完成的**个人实验成果记录**：

| 项目 | 内容 |
|---|---|
| 仓库 | [AIwork4me/yolo26-openvino-rocm-day2-labs](https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs) |
| 课程 | YOLO26 Day 02 — AI 推理加速：OpenVINO × AMD ROCm |
| 基于的提交 | `9ca1ea912dd170cff9c1d6271caea5b0115d5ab1`（`main`，2026-11-08） |
| 许可 | **MIT License**, Copyright (c) 2026 AIwork4me |
| 许可原文 | [`LICENSE`](LICENSE)，从上游仓库逐字取得，未修改一个字 |

MIT 许可允许使用、复制、修改、发布与分发，条件是**在所有副本或实质性部分中保留版权声明和许可声明**。因此本仓库：

- 完整保留上游 `LICENSE` 原文（含其第三方资产段落）；
- 在 [`README.md`](README.md) 与本文明确标注来源与提交号；
- 6 份 notebook 保留课程模板的全部源码、双语说明与 `STUDENT_*` 参数，**没有**删除或改写原有版权/模板内容；
- 不声明对课程模板、课程文档或图表设计的所有权。

## 本仓库原创的部分

- 6 份 notebook 的**执行输出**（设备信息、计时、mAP、图表、预测对照）；
- 每份 notebook 末尾新增的一格分析（标记 `DAY2_SUBMISSION_NOTE`）：预测 vs 实测、原因、代价、证据路径；
- [`README.md`](README.md)、[`RESULTS.md`](RESULTS.md)、本文，以及数据的筛选与组织；
- `results/raw/` 中带 `20261111_*` 时间戳的 20 个实测文件，和 `assets/slides/` 中由这些执行生成的图表。

这些原创内容同样按 MIT 许可提供，与上游许可兼容。

## 第三方资产

以下声明转录自上游 `LICENSE` 原文，适用于课程材料本身；**本仓库不重新分发这些资产**：

- **COCO / Flickr 图像**：按各自原有许可使用（见上游 `docs/DATA_LICENSES.md`）。本仓库不含数据集；`00` 号 notebook 输出中内嵌的样例推理图是执行时的可视化结果。
- **Ultralytics YOLO26 权重**：由 Ultralytics 发布的预训练权重（`ultralytics` 包为 AGPL-3.0）。上游仅在安装时下载，从不随仓库分发；本仓库同样不含 `.pt` / `.bin` 权重文件。
- **OpenVINO / NNCF**：Intel 发行，Apache-2.0。
- **PyTorch ROCm wheels**：AMD 发行。

## 本仓库明确不包含

课程源码（`course/`、`scripts/`、`tests/`）、模型权重、数据集、上游的官方基准结果与示例 CSV。这些请到上游仓库按其 `STUDENT_GUIDE.md` 获取。

## 怎么自行核对数字

`notebooks/` 里每个计时格的输出，都能在 `results/raw/` 找到同时间戳的 `*.csv` 与 `*_summary.json`；`summary.json` 的 `stats.p50_ms` 就是 [`README.md`](README.md) 结果表里的数字（表中为便于阅读做了四舍五入）。05 号实验课程脚本不落 CSV，其数据在 notebook 输出里。
