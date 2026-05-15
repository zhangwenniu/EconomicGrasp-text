# 交接文档：基于文本语义增强的机器人抓取泛化研究

> **目标读者**：接手本项目的 AI Agent（Cursor）。本文档描述项目现状、已完成工作、
> 当前阻塞点，以及接下来需要完成的所有任务。请完整阅读后再动手。

---

## 1. 项目概述

### 研究目标

在 **EconomicGrasp**（ECCV 2024）基础上，通过引入自然语言场景描述，
提升机器人 6-DoF 抓取检测在**未见物体（novel objects）**上的泛化性。

### 核心思路

1. 用豆包视觉大模型为每个场景生成结构化文本描述（物体名称 + 颜色）
2. 用 **CLIP ViT-B/32** 将文本编码为 512 维嵌入，**离线预计算**保存
3. **Stage 1**（已实现）：用场景级文本嵌入通过 **FiLM** 机制条件化 Backbone 点特征
4. **Stage 2**（待实现）：用逐物体文本 Token 与可抓取点特征做 **Cross-Attention**

### 基座模型

**EconomicGrasp**（论文：[arXiv 2407.08366](https://arxiv.org/abs/2407.08366)，
代码：[iSEE-Laboratory/EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp)）

- 稀疏 3D 卷积 TDUnet Backbone（MinkowskiEngine）
- 在 GraspNet-1Billion 数据集上训练
- 相比 SOTA 减少 1/4 训练时间、1/8 显存占用

### 评测数据集

**GraspNet-1Billion**：190 个场景，3 个测试集：

| Split | 场景 ID | 物体类型 |
|---|---|---|
| test_seen | 100–129 | 训练集中出现过的物体 |
| test_similar | 130–159 | 形状相似的新物体 |
| test_novel | 160–189 | 全新类别物体 ← 核心优化目标 |

评估指标：**AP@0.4** 和 **AP@0.8**（抓取成功率，IoU 阈值）

---

## 2. 仓库结构

```
EconomicGrasp-text/
├── dataset/
│   ├── graspnet_dataset.py          # ✏️ 已修改：加载文本特征
│   ├── precompute_text_features.py  # 🆕 新增：离线预计算 CLIP 嵌入
│   ├── generate_economic.py         # 原始（未改动）
│   └── generate_graspness.py        # 原始（未改动）
├── models/
│   ├── economicgrasp.py             # ✏️ 已修改：注入 FiLM
│   ├── modules_economicgrasp.py     # ✏️ 已修改：新增 TextFiLM 类
│   ├── backbone.py                  # 原始（未改动）
│   └── loss_economicgrasp.py        # 原始（未改动）
├── utils/
│   ├── arguments.py                 # 原始（未改动）
│   ├── data_utils.py                # 原始（未改动）
│   ├── label_generation.py          # 原始（未改动）
│   ├── loss_utils.py                # 原始（未改动）
│   └── collision_detector.py        # 原始（未改动）
├── libs/                            # 第三方库（pointnet2, knn, MinkowskiEngine）
├── train.py                         # 原始（未改动，但需要传 use_text 参数，见 §6）
├── test.py                          # 原始（未改动，同上）
├── descriptions.json                # 🆕 豆包 VLM 生成的场景描述（380 条）
├── requirements.txt
└── HANDOFF.md                       # 本文档
```

**数据集目录**（服务器路径，不在仓库中）：
```
./graspnet-dataset/
├── scenes/                          # 原始 RGB-D 图像
├── graspness/                       # 预生成的 graspness 标签
├── economic_grasp_label_300views/   # EconomicGrasp 训练标签
└── text_features/                   # 🆕 待生成：CLIP 文本嵌入（.npz）
    └── scene_xxxx/
        └── kinect/
            └── 0000.npz ~ 0255.npz
```

---

## 3. 已完成的工作（详细）

### 3.1 数据生成：`descriptions.json`

用豆包视觉大模型对 GraspNet **190 个场景** × **2 个相机**（kinect/realsense）
的代表帧图像生成结构化描述，共 **380 条**记录。

格式（key 为图片文件名，value 为物体列表）：
```json
{
  "scene_0000_kinect_0000.png": [
    {"name": "banana",      "color": "yellow"},
    {"name": "screwdriver", "color": "red and black"},
    {"name": "apple",       "color": "red and yellow"}
  ],
  "scene_0000_realsense_0000.png": [...]
}
```

文件位置：仓库根目录 `descriptions.json`（200 KB，已随代码提交）

---

### 3.2 离线预计算脚本：`dataset/precompute_text_features.py`

**作用**：将 `descriptions.json` 中的文本描述用 CLIP 编码为 `.npz` 文件，
每个场景的所有 256 帧都写入同一份嵌入（场景内物体不变）。

**输出格式**（每个 `.npz` 文件）：
```
scene_feat  : float32 [512]         均值池化场景嵌入（供 FiLM 使用）
obj_feats   : float32 [20, 512]     逐物体嵌入，零 padding 到 20 个
obj_mask    : float32 [20]          1=有效, 0=padding
num_objects : int32 scalar          实际物体数
```

**文本构造模板**：`"a {color} {name}"`，例如 `"a yellow banana"`

**运行命令**：
```bash
pip install open_clip_torch
python dataset/precompute_text_features.py \
    --dataset_root ./graspnet-dataset \
    --descriptions descriptions.json
```

**输出路径**：`{dataset_root}/text_features/{scene}/{camera}/{frame:04d}.npz`

**⚠️ 当前阻塞点**：服务器上 `torchvision` 与 `torch` 版本不匹配，
导致 `open_clip_torch` 导入时崩溃。错误信息：
```
RuntimeError: operator torchvision::nms does not exist
```
**修复方法**（在服务器上执行）：
```bash
# 第一步：确认 torch 版本
python -c "import torch; print(torch.__version__)"

# 第二步：根据输出安装对应 torchvision
pip install torchvision==0.20.0  # torch 2.5.x
pip install torchvision==0.19.0  # torch 2.4.x
pip install torchvision==0.18.0  # torch 2.3.x
pip install torchvision==0.17.0  # torch 2.2.x
pip install torchvision==0.16.0  # torch 2.1.x
pip install torchvision==0.15.2  # torch 2.0.x

# 第三步：重跑预计算
python dataset/precompute_text_features.py \
    --dataset_root ./graspnet-dataset \
    --descriptions descriptions.json
```

---

### 3.3 数据集改造：`dataset/graspnet_dataset.py`

在原始文件基础上做了以下修改（**不破坏原有接口**）：

**新增全局常量**（文件顶部）：
```python
_TEXT_FEAT_DIM   = 512
_MAX_OBJ         = 20
_ZERO_SCENE_FEAT = np.zeros(_TEXT_FEAT_DIM,          dtype=np.float32)
_ZERO_OBJ_FEATS  = np.zeros((_MAX_OBJ, _TEXT_FEAT_DIM), dtype=np.float32)
_ZERO_OBJ_MASK   = np.zeros(_MAX_OBJ,                dtype=np.float32)
```

**`__init__` 新增参数**：`use_text=True`

**`__init__` 新增路径列表**（与 `graspnesspath` 结构完全对称）：
```python
self.textpath.append(
    os.path.join(root, 'text_features', x, camera,
                 str(img_num).zfill(4) + '.npz')
)
```

**新增辅助方法** `_load_text_feat(index)`：
```python
def _load_text_feat(self, index):
    """文件不存在时自动回退零向量，训练不中断。"""
    if not self.use_text:
        return _ZERO_SCENE_FEAT.copy(), _ZERO_OBJ_FEATS.copy(), _ZERO_OBJ_MASK.copy()
    path = self.textpath[index]
    if not os.path.exists(path):
        return _ZERO_SCENE_FEAT.copy(), _ZERO_OBJ_FEATS.copy(), _ZERO_OBJ_MASK.copy()
    data = np.load(path)
    return (data['scene_feat'].astype(np.float32),
            data['obj_feats'].astype(np.float32),
            data['obj_mask'].astype(np.float32))
```

**`get_data` 和 `get_data_label` 末尾均新增**：
```python
scene_feat, obj_feats, obj_mask = self._load_text_feat(index)
ret_dict['scene_feat'] = scene_feat    # [512]
ret_dict['obj_feats']  = obj_feats     # [20, 512]
ret_dict['obj_mask']   = obj_mask      # [20]
```

> **注意**：`collate_fn` 无需修改。`scene_feat/obj_feats/obj_mask`
> 均为固定 shape 的 numpy array，会被现有逻辑自动 stack 成 `[B, ...]`。

---

### 3.4 FiLM 模块：`models/modules_economicgrasp.py`

在文件末尾新增 `TextFiLM` 类：

```python
class TextFiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on a scene-level text embedding.
    
    Inputs:
        point_feat : [B, C, N]      backbone point features (C=512)
        scene_feat : [B, text_dim]  mean-pooled CLIP scene embedding
    Output:
        [B, C, N]  modulated point features
    
    初始化为恒等变换（gamma=1, beta=0），不破坏预训练权重加载。
    """
    def __init__(self, point_dim=512, text_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, point_dim),
            nn.ReLU(inplace=True),
            nn.Linear(point_dim, point_dim * 2),  # → gamma [C] + beta [C]
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, point_feat, scene_feat):
        params = self.mlp(scene_feat)               # [B, C*2]
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1) + 1.0           # [B, C, 1]，残差：起始为 1
        beta  = beta.unsqueeze(-1)                  # [B, C, 1]
        return gamma * point_feat + beta
```

---

### 3.5 主模型：`models/economicgrasp.py`

**新增 import**：
```python
from models.modules_economicgrasp import ..., TextFiLM
```

**`__init__` 新增参数**：`use_text=True, text_dim=512`

**`__init__` 新增模块**：
```python
if self.use_text:
    self.text_film = TextFiLM(point_dim=self.seed_feature_dim, text_dim=text_dim)
```

**`forward` 新增注入**（Backbone 输出后、GraspableNet 之前）：
```python
# Stage-1 text conditioning: FiLM on backbone features
if self.use_text and 'scene_feat' in end_points:
    scene_feat = end_points['scene_feat'].cuda().float()  # [B, 512]
    seed_features = self.text_film(seed_features, scene_feat)
```

---

## 4. 网络完整数据流（含文本融合）

```
输入
├── point_clouds        [B, 20000, 3]       点云坐标
├── scene_feat          [B, 512]            CLIP 场景嵌入       ← 文本输入 Stage1
└── obj_feats           [B, 20, 512]        CLIP 逐物体嵌入     ← 文本输入 Stage2（待实现）

TDUnet Backbone
└── seed_features       [B, 512, 20000]

TextFiLM（Stage 1，已实现）
└── seed_features       [B, 512, 20000]     文本条件化后

GraspableNet
├── objectness_score    [B, 2, 20000]
└── graspness_score     [B, 20000]

Mask & FPS
└── seed_features_graspable  [B, 512, 1024]   仅保留可抓取点

ViewNet
└── 选出每点最佳抓取视角（训练时 multinomial 采样，推理时 argmax）

Cross-Attention（Stage 2，待实现）
├── Query: 点特征       [B, 1024, 512 or 256]
├── Key/Value: 文本     [B, 20, 512]
└── 输出: 增强点特征    [B, 1024, 256]

CylinderGrouping
└── group_features      [B, 256, 1024]

GraspHead
├── grasp_angle_pred    [B, 12+1, 1024]
├── grasp_depth_pred    [B, 4+1,  1024]
├── grasp_width_pred    [B, 1,    1024]
└── grasp_score_pred    [B, 6,    1024]
```

---

## 5. 接下来需要完成的任务（按优先级排序）

### 任务 1：修复服务器环境并生成文本特征（阻塞后续所有任务）

```bash
# 在服务器上
python -c "import torch; print(torch.__version__)"
# 根据输出安装对应 torchvision（见 §3.2）

git pull   # 拉取最新代码

python dataset/precompute_text_features.py \
    --dataset_root ./graspnet-dataset \
    --descriptions descriptions.json

# 验证：应该看到类似
# scene_0000/kinect/0000.npz ~ 0255.npz 各 190×2×256 个文件
ls ./graspnet-dataset/text_features/scene_0000/kinect/ | wc -l   # 应输出 256
```

---

### 任务 2：修改 `train.py` 以支持文本特征传参

当前 `train.py` 第 55 行硬编码了模型初始化，未传 `use_text` 参数：
```python
# 当前（原始代码）
net = economicgrasp(seed_feat_dim=512, is_training=True)
```

**需要修改为**：
```python
net = economicgrasp(seed_feat_dim=512, is_training=True, use_text=True)
```

同时需要在 `utils/arguments.py` 末尾（`cfgs = parser.parse_args()` 之前）
新增命令行参数，以便消融实验时方便开关：
```python
parser.add_argument('--use_text', action='store_true', default=False,
                    help='Enable text-conditioned FiLM (Stage 1)')
parser.add_argument('--use_cross_attn', action='store_true', default=False,
                    help='Enable Cross-Attention text fusion (Stage 2)')
```

然后 `train.py` 改为：
```python
net = economicgrasp(seed_feat_dim=512, is_training=True,
                    use_text=cfgs.use_text)
```

---

### 任务 3：启动 Stage 1 训练并记录 baseline 对比

**先跑 baseline（无文本）**：
```bash
python train.py \
    --dataset_root ./graspnet-dataset \
    --camera kinect \
    --log_dir log/baseline \
    --max_epoch 10
```

**再跑 FiLM 版本（有文本）**：
```bash
python train.py \
    --dataset_root ./graspnet-dataset \
    --camera kinect \
    --log_dir log/film_stage1 \
    --max_epoch 10 \
    --use_text
```

**评测命令**：
```bash
# 评测 baseline
python test.py \
    --dataset_root ./graspnet-dataset \
    --camera kinect \
    --checkpoint_path log/baseline/graspness_epoch10.tar \
    --test_mode seen

python test.py ... --test_mode similar
python test.py ... --test_mode novel

# 评测 FiLM
python test.py \
    --dataset_root ./graspnet-dataset \
    --camera kinect \
    --checkpoint_path log/film_stage1/graspness_epoch10.tar \
    --test_mode novel
```

**目标**：FiLM 版本在 `test_novel` 上的 AP@0.4 和 AP@0.8
相比 baseline 有明显提升（test_seen 预期持平或略降）。

---

### 任务 4：实现 Stage 2 Cross-Attention（文本-点云融合）

**位置**：`models/modules_economicgrasp.py`，新增 `TextPointCrossAttn` 类

**设计说明**：

```
输入:
  point_feat : [B, M, D]       M=1024 个可抓取点的特征, D=512 或 256
  obj_feats  : [B, K, 512]     K=20 个物体的文本嵌入（含 padding）
  obj_mask   : [B, K]          有效物体掩码（1=有效, 0=padding）

输出:
  [B, M, D]  文本增强后的点特征
```

**参考实现**（基于现有 `MultiHeadAttn`，代码中已有此类）：

```python
class TextPointCrossAttn(nn.Module):
    """
    让每个可抓取点关注场景内的物体文本描述。
    Query = 点特征，Key/Value = 物体文本 Token。
    """
    def __init__(self, point_dim=256, text_dim=512, n_head=4, dropout=0.1):
        super().__init__()
        self.point_proj = nn.Linear(point_dim, point_dim)
        self.text_proj  = nn.Linear(text_dim,  point_dim)
        self.cross_attn = MultiHeadAttn(dim=point_dim, nhead=n_head, dropout=dropout)
        self.norm       = nn.LayerNorm(point_dim)
        self.ff         = nn.Sequential(
            nn.Linear(point_dim, point_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(point_dim * 2, point_dim),
        )
        self.norm2 = nn.LayerNorm(point_dim)

    def forward(self, point_feat, obj_feats, obj_mask=None):
        """
        point_feat : [B, M, point_dim]
        obj_feats  : [B, K, text_dim]
        obj_mask   : [B, K]  (1=valid, 0=pad)  →  attention mask [B, M, K]
        """
        # 将文本投影到 point_dim 空间
        text_proj = self.text_proj(obj_feats)           # [B, K, point_dim]

        # attention mask: padding 位置设为 0（会被 masked_fill -1e9）
        mask = None
        if obj_mask is not None:
            # [B, 1, K] broadcast → [B, M, K]
            mask = obj_mask.unsqueeze(1)                # [B, 1, K]

        # Cross-attention: 点 Query 文本 K/V
        attn_out = self.cross_attn(
            query=point_feat,
            key=text_proj,
            value=text_proj,
            mask=mask,
        )
        point_feat = self.norm(point_feat + attn_out)   # 残差 + LayerNorm

        # Feed-forward
        point_feat = self.norm2(point_feat + self.ff(point_feat))
        return point_feat
```

**注入位置**（`models/economicgrasp.py`）：

Cross-Attention 应该在 `CylinderGrouping` 之后、`GraspHead` 之前注入：

```python
# 在 forward() 中，group_features 形状为 [B, 256, 1024]

if self.use_cross_attn and 'obj_feats' in end_points:
    obj_feats = end_points['obj_feats'].cuda().float()   # [B, 20, 512]
    obj_mask  = end_points['obj_mask'].cuda().float()    # [B, 20]
    # 转置为 [B, M, C] 做 attention
    gf = group_features.transpose(1, 2)                 # [B, 1024, 256]
    gf = self.text_cross_attn(gf, obj_feats, obj_mask)  # [B, 1024, 256]
    group_features = gf.transpose(1, 2)                 # [B, 256, 1024]

end_points = self.grasp_head(group_features, end_points)
```

需要在 `__init__` 中加：
```python
self.use_cross_attn = use_cross_attn
if self.use_cross_attn:
    self.text_cross_attn = TextPointCrossAttn(
        point_dim=256, text_dim=512, n_head=4
    )
```

同时 `economicgrasp.__init__` 签名改为：
```python
def __init__(self, ..., use_text=True, use_cross_attn=False, text_dim=512):
```

---

### 任务 5：消融实验

| 实验组 | `use_text` | `use_cross_attn` | 说明 |
|---|---|---|---|
| baseline | False | False | 原始 EconomicGrasp |
| +FiLM | True | False | Stage 1 |
| +FiLM+CA | True | True | Stage 1 + Stage 2 |
| +CA only | False | True | 消融 FiLM 贡献 |

每组跑完后用 `test.py` 评测三个 split，填写下表：

| 模型 | seen AP@0.4 | seen AP@0.8 | similar AP@0.4 | similar AP@0.8 | novel AP@0.4 | novel AP@0.8 |
|---|---|---|---|---|---|---|
| baseline | | | | | | |
| +FiLM | | | | | | |
| +FiLM+CA | | | | | | |

---

## 6. 关键代码细节与注意事项

### 6.1 `train.py` 中 `.to(device)` 的问题

`train.py` 第 96–102 行将所有 batch key 移动到 device：
```python
for key in batch_data_label:
    if 'list' in key:
        ...
    else:
        batch_data_label[key] = batch_data_label[key].to(device)
```

`scene_feat`、`obj_feats`、`obj_mask` 都是普通 tensor，会被这段代码自动移到 GPU，
**无需额外处理**。但 `economicgrasp.forward()` 中仍有 `.cuda()` 调用作为保险：
```python
scene_feat = end_points['scene_feat'].cuda().float()
```
这是正确的，保留即可。

### 6.2 文本特征文件缺失时的处理

`_load_text_feat()` 在文件不存在时返回全零向量，此时 FiLM 输出为
`gamma=1, beta=0`（恒等变换），训练不会崩溃。
**但效果等同于无文本**，因此必须先跑完预计算再开始训练。

### 6.3 `obj_mask` 在 Cross-Attention 中的用法

`MultiHeadAttn.attention()` 的 mask 参数：
- shape：`[B, 1, S, L]`（其中 S=query 长度，L=key/value 长度）
- 语义：**0 的位置会被 masked_fill(-1e9)**，即 0=屏蔽，非0=保留

所以 `obj_mask`（1=有效, 0=padding）可以直接传入，
但需要 reshape 到 `[B, 1, 1, K]` 才能与 `[B, nhead, M, K]` 广播：
```python
mask = obj_mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, K]
```

### 6.4 关于 `descriptions.json` 的 key 格式

```
"scene_0000_kinect_0000.png"
 ──────  ────  ──────  ────
 固定    编号   相机    帧号
```

解析逻辑在 `precompute_text_features.py` 的 `parse_key()` 函数中：
```python
parts = key.replace(".png", "").split("_")
scene  = f"scene_{parts[1]}"    # "scene_0000"
camera = parts[2]               # "kinect"
frame  = parts[3]               # "0000"
```

### 6.5 关于 `MAX_OBJ=20` 的选择

GraspNet 场景中通常有 5–15 个物体，20 作为上界足够。
如果某场景超过 20 个物体，`precompute_text_features.py` 会截断（取前 20 个）。

### 6.6 CLIP 嵌入的 L2 归一化

预计算时已对每个物体嵌入做了 L2 归一化：
```python
feats = feats / feats.norm(dim=-1, keepdim=True)
```
`scene_feat`（均值池化）**未做二次归一化**。如需对齐分布，可以在
`_load_text_feat()` 中加：
```python
scene_feat = scene_feat / (np.linalg.norm(scene_feat) + 1e-8)
```
目前暂未添加，可在后续实验中对比有无归一化的影响。

---

## 7. 环境配置

### 服务器 conda 环境

```bash
conda activate ecograsp   # 已有的训练环境
```

已安装（确认可用）：
- PyTorch（版本待确认）
- MinkowskiEngine
- pointnet2（需编译：`cd libs/pointnet2 && python setup.py install`）
- knn（需编译：`cd libs/knn && python setup.py install`）
- GraspNetAPI（用于评测）

需要额外安装：
```bash
pip install open_clip_torch   # 用于预计算文本特征（修复 torchvision 后）
```

### 代码同步

```bash
# 服务器每次拉取最新代码
git pull origin main
```

---

## 8. 快速上手检查清单

按顺序执行：

- [ ] `git pull` 确认代码最新
- [ ] `python -c "import torch; print(torch.__version__)"` 确认 torch 版本
- [ ] 安装对应 torchvision
- [ ] `pip install open_clip_torch` 验证安装成功：`python -c "import open_clip; print('ok')"`
- [ ] 运行预计算：`python dataset/precompute_text_features.py --dataset_root ./graspnet-dataset --descriptions descriptions.json`
- [ ] 验证输出：`ls ./graspnet-dataset/text_features/scene_0000/kinect/ | wc -l` 应为 256
- [ ] 修改 `train.py` 和 `utils/arguments.py`（见任务 2）
- [ ] 跑 baseline 训练（`--use_text` 不加）
- [ ] 跑 FiLM 训练（加 `--use_text`）
- [ ] 对比 test_novel AP 数字
- [ ] 实现 Stage 2 Cross-Attention（见任务 4）
- [ ] 跑消融实验，填写结果表格（见任务 5）

---

## 9. 联系上下文

本文档由 Claude（claude-sonnet-4-5）与项目负责人协作产出，
对应 GitHub 仓库的初始提交（commit `ae64f3b`）。
如有疑问，参考 `HANDOFF.md` 提到的各文件的具体代码实现。
