# YOLO26 Day 02 · 推理加速实测

**AI Inference Acceleration on AMD CPU + GPU — six measured labs, one variable at a time.**

在 AMD EPYC 9334 + Radeon PRO W7900D（ROCm）上，用同一份官方 `yolo26l.pt` 权重跑完 6 组推理加速实验。每组**只改一个变量**，同时记录四件事：快了多少、占多少内存/显存、精度掉没掉、延迟口径有没有被偷换。所有数字都来自 notebook 自己的执行输出，附原始 CSV / JSON 与图表。

> 这不是一次「跑分」。三个实验的结论是**没变快**或**拿延迟换吞吐**，照样原样记录。

---

## 一眼看懂结果

| # | 实验 | 基线 → 方案 | 实测 | 结论 |
|---|---|---|---|---|
| 00 | 环境核验 | — | 权重 sha256 对上、样例图检出 `person` / `skis` | 不计时，只确认机器与模型是真的 |
| 01 | CPU vs GPU | 200.588 ms → **13.103 ms** | **15.31×** | 我猜 100×，错得远；3.0 ms 固定开销卡住上限 |
| 02 | OpenVINO on CPU | 136.296 ms → **50.998 ms** | **2.67×** | mAP 0.6398 → 0.6349，几乎没掉 |
| 03 | INT8 量化 | 99.9 MB / 25.094 ms → **26.4 MB / 19.109 ms** | 小 **3.79×**、快 **1.31×** | mAP 0.6349 → 0.5682（掉 0.0667） |
| 04A | 缩分辨率 | imgsz 640 12.659 ms → 512 **12.671 ms** | **没有变快** | 精度反而 0.6398 → 0.6159 |
| 04B | 加大 batch | 吞吐 109.9 → **147.6 img/s** | **1.34×** | 单批延迟 9.098 → 54.212 ms，显存到 0.98 GB |
| 05 | CPU+GPU 流水线 | 77.1 → **89.6 img/s** | **1.16×** | 60/60 张结果完全一致；单张 P50 13 → 346 ms |

完整推理过程、代价分析与证据路径见 [RESULTS.md](RESULTS.md)。

**三条贯穿全部实验的结论**

1. **Amdahl 定律出现三次**：01 号固定开销卡住 GPU 加速上限、04 号缩图省下的算力被固定开销吃光、05 号流水线收益上限只有约 17%。
2. **「拿延迟换吞吐」出现三次**：02 号 THROUGHPUT、04 号 batch=8、05 号队列深度 4。吞吐涨的代价都是单张等更久。
3. **别偷换口径**：`equivalent 6.8 ms/img` 是平摊值不是请求延迟；THROUGHPUT 的 108.2 ms 是 model-only，与端到端 51.0 ms 不能相减。

---

## 实测环境

以下来自 `notebooks/00_AMD_Environment_and_YOLO26l.ipynb` 的真实输出，不是从课程 README 抄的：

| 项目 | 实测值 |
|---|---|
| CPU | `AMD EPYC 9334 32-Core Processor`，64 物理核 / 128 线程，2 NUMA；自检 ISA `avx2` / `avx512f` / `avx512_bf16` |
| GPU | `AMD Radeon Pro W7900D`，`gfx1100` |
| PyTorch | `2.12.0+rocm7.14.1`，HIP `7.14.60850` |
| OpenVINO | `2026.4.1-22982-e213a147257`（CPU plugin，oneDNN） |
| Ultralytics | `8.4.210` |
| 模型 | `yolo26l.pt`，53.2 MB，sha256 `9fe3c544…db26`，80 类 COCO 检测 |
| 验证子集 | 120 张图 |

运行平台是 AMD Radeon Cloud 的托管容器实例。notebook 输出里的 `/workspace/yolo26-openvino-rocm-day2-labs` 就是当时的仓库路径。

---

## 仓库里有什么

```
README.md            本文件
RESULTS.md           六组实验的预测 / 实测 / 原因 / 代价 / 证据
ATTRIBUTION.md       课程来源、许可与第三方资产声明
LICENSE              上游课程仓库的 MIT 许可原文（逐字保留）
notebooks/           6 份已完整执行、带全部输出的 notebook
                     每份末尾有一格「预测 vs 实测 + 一段分析」
results/raw/         20 个自测文件：5 组 × 2 变体 × (1 CSV + 1 summary JSON)
assets/slides/       7 张 PNG + 7 张 SVG 图表
```

| Notebook | 代码格 | 输出 | 报错 |
|---|---:|---:|---:|
| `00_AMD_Environment_and_YOLO26l.ipynb` | 7 | 7 | 0 |
| `01_Lab_A_CPU_vs_GPU_Baseline.ipynb` | 9 | 9 | 0 |
| `02_Lab_B_OpenVINO_AMD_CPU.ipynb` | 7 | 7 | 0 |
| `03_Lab_C_OpenVINO_INT8.ipynb` | 9 | 9 | 0 |
| `04_Lab_D_ROCm_GPU_Optimization.ipynb` | 6 | 6 | 0 |
| `05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb` | 7 | 7 | 0 |

---

## 怎么复现

本仓库只放**我跑出来的成果**：6 份已执行的 notebook、实测数据与图表。课程的运行时代码（`course/`、`scripts/`、`tests/`）、模型权重和数据集都不在这里，由上游仓库的安装脚本准备。

1. 取上游课程仓库：[AIwork4me/yolo26-openvino-rocm-day2-labs](https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs)（本成果基于 `main` 的 `9ca1ea912dd170cff9c1d6271caea5b0115d5ab1`）。
2. 按上游 `docs/STUDENT_GUIDE.md` 与 `scripts/setup.sh` 准备 ROCm 环境；它会下载官方权重、验证图与校准图，并注册 kernel `Python 3.12 (YOLO26 Day 02 · ROCm)`。
3. 用该 kernel 逐格运行 6 份 notebook。本仓库的 `notebooks/` 可直接替换上游同名文件，用来对照输出。
4. 每次运行会在 `results/raw/` 生成带时间戳的 CSV + `summary.json`；图表写到 `assets/slides/`。

**硬件前提**：需要 AMD ROCm 可用的 GPU（本成果在 gfx1100 / W7900D 上测得）。换成别的卡，绝对数字会变，结论里的倍数关系也可能变。

---

## 数据口径与诚实性说明

这几条特意写清楚，免得数字被误读：

- **测量规格**：本成果每个变体是 **10 次迭代取 P50**（见各 `summary.json` 的 `iters: 10`），与上游 v2.0.1 官方基准「3 轮 × 50 样本、取各轮 P50/P95 的中位数」**不是同一规格**，两组数字不可混用或互相印证。
- **课程示例数据一律剔除**：上游仓库自带的 2026-11-08 示例 CSV、官方 v2.0.1 结果表 `RESULTS_TABLE.md`、以及基准脚本生成的汇总 `latest.json`，都**没有**收进本仓库。收进来的 20 个文件全部能与本仓库 notebook 的执行输出逐条对应。
- **00 / 01 / 02 的执行方式**：云端实例被平台回收导致这三份输出丢失，随后用 `jupyter nbconvert --execute` 在同一环境对 notebook 本体批量复现补齐；03 / 04 / 05 是在 JupyterLab 中逐格执行跑出来的。两者都是 notebook 自己的真实输出，但执行方式不同，如实说明。
- **04 / 05 的「预测」**：课程模板没有给这两份设 `my_*` 预测变量，格子里的「我本以为」是**跑完后补记的当时预期**，不是跑前写进 notebook 的。这两份里已经写明。00 / 01 / 02 / 03 的预测变量原本就在模板里，且有打印输出为证。
- **Lab E 没有 CSV**：05 号课程脚本不落盘原始数据，吞吐、延迟、一致性、CPU/GPU 分段时间都在 notebook 输出里，`06_serial_vs_async_pipeline.png` 是对应图表。
- **两处不能相减**：02 号 `108.2 ms` 是纯 Runtime + 随机输入的 model-only 口径；04 号 `equivalent 6.8 ms/img` 是 batch 平摊值，均非端到端单请求延迟。
- **课程改动**：为让环境跑通，云端对课程脚本做过 3 处最小改动（下载源换镜像、补齐漏掉的 CLI 参数、中文字体优先级），均留备份、未回推上游、不影响测量逻辑。本仓库不含这些改动。

---

## English summary

Six hands-on inference-acceleration labs for the official YOLO26l detector, executed on an AMD EPYC 9334 + Radeon PRO W7900D (ROCm) cloud instance: CPU/GPU baseline with Amdahl analysis, PyTorch → OpenVINO on CPU, INT8 quantization, resolution and batch tuning on GPU, and a CPU+GPU async pipeline. Every reported number traces to an executed notebook cell and a per-iteration CSV/`summary.json`; course-provided sample results and a separate background benchmark were deliberately excluded. The interesting part is that three of the six **did not** deliver a free speedup — fixed overhead, accuracy loss, or latency-vs-throughput trade-offs dominate.

## License & attribution

Course material is © 2026 AIwork4me, MIT licensed; the original license text is kept verbatim in [LICENSE](LICENSE). See [ATTRIBUTION.md](ATTRIBUTION.md) for the upstream commit, third-party asset notices, and what is original to this repository.
