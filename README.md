# MRL 多模态检索（验证阶段）

把 MRL 引入 CLIP 图文检索：训一个 `512->512` 残差 Adapter，用嵌套 InfoNCE 让 embedding 前缀可独立做低维检索，降低延迟/存储。
**完整方法与全部实验数字见 [`RESULTS.md`](RESULTS.md)。**

## 环境（Step 0，已完成）

```powershell
conda create -n mrl-retrieval python=3.12 -y
conda activate mrl-retrieval
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/00_check_env.py     # cuda 应为 True、依赖齐全
```

现状：Python 3.12.13 / torch 2.6.0+cu124（`cuda_avail=True`, RTX 4070 Laptop）/ open_clip 3.3.0 / faiss 1.14.3。

HF 缓存已用 `HF_HOME=D:\hf-cache` 挪到 D 盘（一个变量同时管仓库下载 `hub/` 与 arrow 缓存 `datasets/`）。

## 运行顺序（在项目根目录执行）

```
python scripts/00_fetch_data.py    # 取 Flickr30k 到 data_root（4.4GB，一次性；断点续传）
python scripts/01_sanity_check.py  # 复现官方 baseline：i2t R@1 86.30 / t2i R@1 69.84
python scripts/02_extract_features.py                                                # 抽特征缓存（地基）
python scripts/03_eval_baseline_pca.py                                               # ①② 零训练出曲线
python scripts/04_train_adapter.py                                                   # ④ adapter_mrl
python scripts/04_train_adapter.py --set loss.nested=false --set experiment.name=adapter_plain  # ③ 消融
python scripts/05_evaluate_all.py                                                    # 四组对比表+曲线
```

## 可选实验

```
python scripts/07_ablate_loss.py                       # τ × c_m 的 3×3 消融
python scripts/08_cascade.py --gallery full31k         # 级联检索（含训练图，仅看保持率）
python scripts/09_fetch_coco.py                        # 取 COCO val2014（6.65GB，干净干扰池+跨域基准）
python scripts/01_sanity_check.py --dataset coco       # COCO 5k test 基准：i2t 59.44 / t2i 42.31
python scripts/10_extract_coco.py                      # 抽 COCO 特征
python scripts/08_cascade.py --gallery cross_domain    # 干净大图库（31,504）上的级联

# 同域干净图库需先另训一对 *_ho（留出 6000 图不参与训练）
python scripts/04_train_adapter.py --set data.use_holdout=true --set experiment.name=adapter_mrl_ho
python scripts/04_train_adapter.py --set data.use_holdout=true --set loss.nested=false --set experiment.name=adapter_plain_ho
python scripts/08_cascade.py --gallery same_domain
```

## 数据

**Flickr30k**：`nlphuji/flickr30k` → `{data_root}/flickr_annotations_30k.csv`（含 Karpathy `split` 列）+ `{data_root}/images/*.jpg`。
Karpathy split 实测：train 29,000 / val 1,014 / test 1,000，每图恰好 5 条 caption。

**MS-COCO**：`{coco_root}/karpathy_coco.csv` + `{coco_root}/images/*.jpg`（只下 val2014）。
池子切开**永不相交**：restval 30,504 = 干扰池（永不训练）/ test 5,000 = 跨域基准 / validation 5,000 = 跨域调参 / train 82,783（在 train2014，未下）= Phase 2 LoRA 训练池。

两者落地后**全程读磁盘、不走网络**。

## 结构

- `configs/` 唯一配置源（改 yaml，不改代码）
- `src/` 库（单一职责模块）
- `scripts/` 编号入口（薄壳，只编排）
- `outputs/` 产物（features / checkpoints / results / figures）

`src/` 全部实装、无 stub：`config` `data` `model` `features` `adapter` `losses` `reducers`
`metrics` `retrieval` `evaluate` `train` `lora` `cascade`。

## 当前进度（2026-07-26）

完整结论、图表与注意事项见 [`RESULTS.md`](RESULTS.md)。一句话版：

| 阶段 | 状态 | 核心数字 |
| --- | --- | --- |
| 地基（两个数据集的 sanity check） | ✅ | Flickr30k i2t 86.30 / t2i 69.84、COCO 59.44 / 42.31，**均逐位复现官方值** |
| Phase 1 四组对比 | ✅ | m=64 保住满维 95%，延迟降 94%；③④ 隔离出嵌套净贡献 +24.7(t2i)/+26.1(i2t) @m=32 |
| 损失消融（τ × c_m 3×3） | ✅ | τ 必须可学（收敛≈0.055）；`per_dim` 无用可砍；`c=inverse` 低维最好 |
| 级联检索（3 个图库） | ✅ | MRL 22.4× 无损，原生截断连 99% 都到不了；**MRL ≈ 次优的 2 倍**在三个图库上一致 |
| 跨域评测（COCO 检索任务） | ✅ | 低维可迁移（m=32 i2t 36.40 vs 冻结 4.44）；**满维域外亏 5~6 个点** |
| Phase 2 LoRA | ✅ | 解冻净贡献全维度 +2~3 个点（域内域外都涨）；1.31% 参数、32 分钟 |
| 蒸馏正则扫描 | ✅（负结果） | **正则错了对象** —— 缺口是 adapter 域特化，不是骨干漂移（骨干余弦仍 0.98） |
| 下一步 | 待定 | 见方案 §十bis 的三条路 + 加数据分期方案 |

**checkpoints 现存 6 个**：`adapter_{mrl,plain}.pt`（Phase 1 主结果）、`adapter_{mrl,plain}_ho.pt`
（留出 6000 图版，同域干净级联专用）、`ctrl_adapter_only.pt`（LoRA 对照组）、`lora_mrl.pt`（Phase 2）。
消融与蒸馏扫描的 12 个 checkpoint 已清理 —— 结论都在文档里，重跑 `07`/`06` 即可复现。

## 免代理下载

`huggingface.co` 直连会被重置、`images.cocodataset.org` 直连只有 0.01 MB/s（等于不可用），
但 **`hf-mirror.com` 直连可用（实测 ~1.5 MB/s，与走代理的 1.46 MB/s 同一水平）**：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 之后所有 HF 下载（含 00_fetch_data.py、CLIP 权重）都免代理
```
