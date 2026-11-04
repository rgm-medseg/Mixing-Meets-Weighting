# CoMixW 关键实验备份 · 2026-11-04

本目录保存 5 个主实验和关键消融/对照的清单、代码、脱敏日志、配置、指标原始表及恢复工具。大型权重放在本次独立 Release；在 API 登录完成前，附件仅保留本地，链接标为计划地址，不表示已上传。

原训练目录只读，未删除文件、未修改训练逻辑、未启动训练或评估。权重均在稳定后复制并核对 SHA-256。`PAPER_AUTHOR_CONFIRMED_FACTS.md` 的固定结果优先于旧历史记录：COVID 主 Full HD95 为 22.79/11.88 mm；原始 ACDC10 Full 为 90.35%，控制信号替换为 89.98%。

| 主实验 | 论文最佳步数 | 论文 Dice (%) | 最后训练状态 | 日志/配置 |
|---|---:|---:|---|---|
| acdc5 | 28400 | 88.7674 | 30000；TRAINING_STATE_PRESENT_WITH_WORKER_RNG_LIMITATION | 已保存；见 experiments.json |
| acdc10 | 29600 | 90.3497 | 30000；TRAINING_STATE_PRESENT_WITH_WORKER_RNG_LIMITATION | 已保存；见 experiments.json |
| promise20 | 24400 | 83.8640 | 30000；TRAINING_STATE_PRESENT_WITH_SAMPLER | 已保存；见 experiments.json |
| covid5 | 22500 | 67.3144 | 30000；TRAINING_STATE_PRESENT_WITH_SAMPLER | 已保存；见 experiments.json |
| covid10 | 9000 | 67.9408 | 30000；TRAINING_STATE_PRESENT_WITH_SAMPLER | 已保存；见 experiments.json |

`experiments.csv/json`：实验、比例、种子、模块、验证/测试来源、训练日志、配置、实际源代码和恢复状态。`weights.csv/json`：原始权重路径、对象 SHA、大小、附件、实验引用和 CPU 检查。`files.json`：各文本/标量文件的来源、原 SHA、备份 SHA、压缩方式及实验关联。共享历史材料有明确标记，不能当作每个实验实际使用它的证明。

`reports/checkpoint_audit.json` 和 `reports/model_compatibility.json` 分别记录 CPU 反序列化与主实验严格模型结构加载；`reports/tensorboard_audit.json` 记录事件内容检查。文件校验、模型加载、论文指标复现是三个独立状态。本次没有重跑评估，不声称复现了论文指标。

原训练目录没有 Git 历史；出版基线 commit 和可取得的差异见 `reports/git_provenance.json`。历史依赖、初始化生成记录、未完成候选和恢复限制见 `reports/limitations.json`。所有原始医学数据须从数据提供方另行取得。

## 新服务器下载、验证和恢复文件

以下命令以仓库根目录为起点；仅在 Release 上传成功后，`--download` 才能取得附件。当前可从原备份服务器复制 `assets/` 目录使用相同命令验证恢复。

```bash
git clone --branch backup/comixw-critical-2026-11-04 https://github.com/rgm-medseg/Mixing-Meets-Weighting.git
cd Mixing-Meets-Weighting
COMIXW_CATALOG="$PWD/backups/2026-11-04"
python "$COMIXW_CATALOG/tools/restore_backup.py" --catalog "$COMIXW_CATALOG" --assets "$PWD/backup-assets" --download --verify-only
python "$COMIXW_CATALOG/tools/restore_backup.py" --catalog "$COMIXW_CATALOG" --assets "$PWD/backup-assets" --destination "$PWD/restored"
```

`restore_backup.py` 验证全部附件 SHA 后解包，再核对每个对象 SHA，恢复原相对路径。重复权重使用硬链接引用同一对象。文本中的大日志/CSV 以确定性 gzip 保存，恢复时自动解压。当前各附件均小于 2 GiB，无须分卷拼接。

## 环境与评估

原环境为 Python 3.10、PyTorch 2.4.1+cu121；历史及本次环境快照均已保存。先安装匹配硬件的 PyTorch 和出版代码依赖，再准备原划分与预处理格式的数据。

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r restored/publication/requirements-core.txt
export DATA_ROOT=/datasets
cd restored/publication
python tools/evaluate_paper.py --config configs/paper/acdc5.json --checkpoint ../project/experiments/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3/checkpoints/best.ckpt --data-root "$DATA_ROOT" --output-dir /work/eval/acdc5
python tools/evaluate_paper.py --config configs/paper/acdc10.json --checkpoint ../project/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt --data-root "$DATA_ROOT" --output-dir /work/eval/acdc10
python tools/evaluate_paper.py --config configs/paper/promise20.json --checkpoint ../project/experiments/three_innovation_ablation_transfer_v22_0/PROMISE20_FINAL3/self_train/unet_best_model.pth --data-root "$DATA_ROOT" --output-dir /work/eval/promise20
python tools/evaluate_paper.py --config configs/paper/covid5.json --checkpoint ../project/experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/self_train/unet_best_model.pth --data-root "$DATA_ROOT" --output-dir /work/eval/covid5
python tools/evaluate_paper.py --config configs/paper/covid10.json --checkpoint ../project/experiments/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3/self_train/unet_best_model.pth --data-root "$DATA_ROOT" --output-dir /work/eval/covid10
```

`DATA_ROOT` 下应有 `ACDC/`、`PROMISE12/`、`COVID19_OFFICIAL20/`；具体预处理和数据来源见 `records/publication/README.md`、`tools/prepare_data.py` 及保留的原脚本。上述评估命令未在本次任务执行；请将新结果另存，不覆盖原始 receipt。

## 恢复训练

五组最后状态均为 30000 步，已完成原预算。下面工具默认选择仍有原预算更新的已保存状态：ACDC 验证最佳完整状态；PROMISE/COVID 第 25000 步完整状态。先打印实际 argv、初始化复制和数据链接计划，再明确加入 `--execute` 执行。

```bash
cd /work/Mixing-Meets-Weighting
COMIXW_CATALOG="$PWD/backups/2026-11-04"
python "$COMIXW_CATALOG/tools/resume_main.py" --catalog "$COMIXW_CATALOG" --restored "$PWD/restored" --setting covid5 --data-root /datasets --output-root /work/resumed
python "$COMIXW_CATALOG/tools/resume_main.py" --catalog "$COMIXW_CATALOG" --restored "$PWD/restored" --setting covid5 --data-root /datasets --output-root /work/resumed --execute
```

`--setting` 可选 `acdc5/acdc10/promise20/covid5/covid10`。工具使用保留的历史训练入口、原 run identity 和原 30000 步预算，并检查二分类训练核心、初始化与清单 SHA；输出必须是新目录。ACDC 缺少完整 worker RNG/prefetch 状态，不能承诺逐步一致。恢复工具的参数构造和 CPU 状态已检查，本次没有执行训练恢复。

## Release 上传重试

SSH 只能推送 Git，创建/上传 Release 还需要 GitHub API 登录。工具从 `GH_TOKEN/GITHUB_TOKEN`、GitHub CLI 登录或用户私有凭据文件读取；凭据不进入本目录。`--login` 会在终端隐藏输入并保存到备份外的权限 600 文件。

```bash
python backups/2026-11-04/tools/upload_release.py --catalog backups/2026-11-04 --assets /path/to/assets --download-cache /path/to/remote-verification --login
```

上传工具核对账号与仓库写权限、远端备份 commit 和 tag；不覆盖已有附件。全部附件上传后重新从 GitHub 下载逐项核对 SHA，并实测解包恢复。结果写入 `reports/remote_verification.json`；存在失败即报告并保留已上传附件和本地包，不购买存储服务。
