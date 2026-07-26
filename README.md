# MRL-Retrieval

把 **Matryoshka Representation Learning（MRL，套娃表示学习）** 用到 CLIP 图文检索上：训一个
1.05M 参数的残差 adapter，用**嵌套 InfoNCE** 让 embedding 的**前缀本身就是可用的检索表示**，
从而按需截断到低维、直接降低检索延迟与存储 —— 不必为每个目标维度单独训一个模型。

**64 维即保住满维 95% 的召回，单条 query 延迟降 94%；级联检索能在保住 ≥99% 精确检索召回的前提下加速 22×。**

完整方法、全部数字表与图见 **[RESULTS.md](RESULTS.md)**。

![recall vs dim](outputs/figures/r1_vs_dim.png)

---

## 核心做法

对每对图文，adapter 输出 512 维向量 `e`；对每个档位 `m ∈ {32,64,128,256,512}`，
**先取前缀 `e[:m]`、再 L2 归一化**，算对称 InfoNCE，最后按权重 `c_m` 求和：

```
L = Σ_m  c_m · InfoNCE( normalize(e_img[:m]), normalize(e_txt[:m]), τ )
```

> ⚠️ 顺序不能反。先归一化整段再切前缀会得到非单位长度的向量、召回会静默算错。
> `src/metrics.py` 里有单位长度断言专门拦这个。

四组对比（同一张 R@1 × 维度曲线上比）：

| | 名称 | 做法 | 是否训练 |
| --- | --- | --- | --- |
| ① | Truncate | 原生 CLIP embedding 直接截断到 m | 否 |
| ② | PCA | 在 train 上拟合 PCA（图文共用一个投影） | 否 |
| ③ | Adapter-plain | adapter **只在满维**算 InfoNCE，评测时截断 | 是 |
| ④ | **Adapter-MRL** | adapter 用**嵌套**损失训练 | 是 |

③ 的作用是隔离"嵌套"本身的贡献：③④ 的架构、参数量、数据、epoch、学习率、随机种子、
温度模式、best-checkpoint 选择准则全部相同，**唯一差别是损失里有 1 项还是 5 项嵌套项**。
在 m=32 处 ④ 比 ③ 高 +24.7（t2i）/ +26.1（i2t）个点。

---

## 环境

需要一块支持 CUDA 的显卡。全流程实测可在 **8 GB 显存**内跑完（含 Phase 2 的骨干微调）。

```bash
conda create -n mrl-retrieval python=3.12 -y
conda activate mrl-retrieval
# torch 按自己的 CUDA 版本从官方 index 装，下面是 cu124 的例子
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/00_check_env.py          # 检查 CUDA 可用性与依赖
```

参考版本（本仓库结果所用）：Python 3.12 · torch 2.6.0+cu124 · open_clip 3.3.0 · faiss-cpu 1.14.3。

## 数据

两份数据都由脚本下载到本地，**之后全程读磁盘、运行期不联网**。
下载位置由 `configs/default.yaml` 的 `paths.data_root` / `paths.coco_root` 指定，**换机器只需改这两行**。

| 数据集 | 内容 | 下载量 | 脚本 |
| --- | --- | --- | --- |
| **Flickr30k** | Karpathy split：train 29,000 / val 1,014 / test 1,000，每图 5 条 caption | 4.4 GB | `00_fetch_data.py` |
| **MS-COCO**（可选） | Karpathy split 的 val2014 部分：test 5,000 / val 5,000 / restval 30,504 | 6.6 GB | `09_fetch_coco.py` |

COCO 只有跨域实验用得到（第 5、6 节）。它的 `train`（82,783 张，在 `train2014.zip` 里）
**本仓库不下载**，被刻意保留为"永不当干扰项"的训练池。

## 运行

```bash
# ---- 基础流程（只需 Flickr30k）----
python scripts/00_fetch_data.py         # 下数据（一次性，支持断点续传）
python scripts/01_sanity_check.py       # 复现官方 baseline：i2t 86.30 / t2i 69.84
python scripts/02_extract_features.py   # 抽并缓存冻结 CLIP 特征（后续训练全靠它，秒级迭代）
python scripts/03_eval_baseline_pca.py  # ①② 零训练，直接出两条曲线
python scripts/04_train_adapter.py      # ④ Adapter-MRL
python scripts/04_train_adapter.py --set loss.nested=false --set experiment.name=adapter_plain   # ③ 消融
python scripts/05_evaluate_all.py       # 四组汇总表 + 曲线图
```

**先让 `01_sanity_check.py` 通过再往下走。** 它核对的是 OpenCLIP 官方公布的检索数字；
对不上说明预处理或 i2t/t2i 协议有问题，此时后面所有对比都无意义。

```bash
# ---- 可选实验 ----
python scripts/07_ablate_loss.py                       # 温度 τ × 权重 c_m 的 3×3 消融
python scripts/08_cascade.py --gallery full31k         # 级联检索
python scripts/06_train_lora.py                        # Phase 2：LoRA 解冻骨干
python scripts/06_train_lora.py --name ctrl --set "lora.towers=[]"   # 对照组：同预算但骨干仍冻结

# 需要 COCO
python scripts/09_fetch_coco.py
python scripts/01_sanity_check.py --dataset coco       # 第二道 baseline：i2t 59.44 / t2i 42.31
python scripts/10_extract_coco.py
python scripts/08_cascade.py --gallery cross_domain    # 干净的大图库（31,504）上的级联
python scripts/11_eval_cross_domain.py                 # 换成 COCO 检索任务，看跨域迁移

# 同域干净图库：需先另训一对留出版 adapter（留出的图不参与训练，否则干扰项不干净）
python scripts/04_train_adapter.py --set data.use_holdout=true --set experiment.name=adapter_mrl_ho
python scripts/04_train_adapter.py --set data.use_holdout=true --set loss.nested=false --set experiment.name=adapter_plain_ho
python scripts/08_cascade.py --gallery same_domain
```

所有超参都在 `configs/default.yaml` 里，命令行用 `--set key.path=value` 临时覆盖，**不需要改代码**。

### 耗时参考（RTX 4070 Laptop 8GB）

| 步骤 | 耗时 |
| --- | --- |
| 抽特征（31k 图 + 155k caption） | ~4 分钟 |
| 训 adapter（冻结骨干，20 epoch） | ~1 分钟 |
| 3×3 损失消融（9 组） | ~20 分钟 |
| LoRA（5 epoch，batch 256） | ~32 分钟 |

冻结骨干阶段之所以只要 1 分钟，是因为特征已缓存、训练只在 512 维向量上跑。
LoRA 阶段骨干每步都变，**无法复用缓存**，必须整套前向过编码器，所以贵一个量级。

---

## 仓库结构

```
configs/default.yaml    唯一配置源（改 yaml，不改代码）
src/                    库，单一职责模块
  data / model / features      数据加载、冻结 CLIP、特征缓存
  adapter / losses             MRLAdapter、嵌套 InfoNCE（含 τ 与 c_m 的各种模式）
  reducers                     统一降维接口，把四组抹平成同一个 transform(emb, m)
  metrics / retrieval          双向 R@K；faiss 索引与延迟/存储测量
  cascade                      两阶段检索（低维粗筛 + 满维重排）
  lora                         手写 LoRA 注入
  evaluate / train             评测编排、训练循环
scripts/                入口，编号即执行顺序，只做编排
outputs/
  results/  figures/           已随仓库提交，是结论的证据
  features/  checkpoints/      运行时生成，未提交（体积大且可重算）
```

`outputs/features/` 与 `outputs/checkpoints/` 不在版本控制内，脚本会自动创建。
克隆后按上面的顺序跑一遍即可重新生成。

## 已知限制

- **单个随机种子，没有误差棒。** 报告的 2~3 个点的效应相对于 1000 图测试集的常见波动是显著的，
  但没有做多种子复现。
- **延迟测量有约 20% 的跨运行噪声**（同一个 256 维配置在不同运行中测到 1.25~1.51 ms）。
  只在同一次运行内比较维度间的缩放，不要跨运行比绝对值。
- **级联的召回是在 31k 图库上算的**，与文献里 1000 图测试集的数字**不可比**，
  只用于同一图库内各条件之间的比较。
- **满维的跨域能力有回退**：adapter 在 Flickr30k 上涨 0.3~1.0 个点，但在 MS-COCO 上掉 4.3~6.0 个点。
  低维（m≤128）跨域依然大幅领先，但如果你需要的是"满维直接替换原 CLIP"，这个 adapter 不合适。
  原因已定位到 adapter 的域特化而非骨干漂移，详见 [RESULTS.md](RESULTS.md) 第 7 节。

## 如果 Hugging Face 下载失败

数据与模型权重都走 Hugging Face。若你所在网络访问 `huggingface.co` 不稳定，可切到镜像端点
（该变量被 `huggingface_hub` 直接支持，不需要改代码）：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # Windows PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
```

另外 `HF_HOME` 可用来把模型/数据缓存挪到空间更大的盘：`export HF_HOME=/path/to/cache`。

## 引用的资源

- 骨干：[OpenCLIP](https://github.com/mlfoundations/open_clip) ViT-B/16，`laion2b_s34b_b88k`
- Flickr30k（Karpathy split）：[`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k)
- MS-COCO（Karpathy split 标注）：[`yerevann/coco-karpathy`](https://huggingface.co/datasets/yerevann/coco-karpathy)，图像取自 [cocodataset.org](https://cocodataset.org)
- 方法出处：Kusupati et al., *Matryoshka Representation Learning*, NeurIPS 2022

## License

MIT，见 [LICENSE](LICENSE)。
