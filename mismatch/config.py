"""
实验配置。所有扫描维度集中在这里, 三个入口脚本共用。

针对 trn2.3xlarge (单颗 Trainium2, 8 个 NeuronCore-v3, 96 GiB HBM) 设计。
换到 48xlarge 时只需要改 TP_DEGREES 和 LNC_CONFIGS 的取值, 代码不用动。
"""

from dataclasses import dataclass, asdict
import hashlib
import json


# --- 固定不变的量 ----------------------------------------------------------

N_PROMPTS = 512          # 测试集大小。512 足够让分布的 p99 稳定
SEED = 20260903
MAX_NEW_TOKENS = 512     # 前向实验不需要很长, 512 够看清失配随位置的累积

# 采样参数: 绝对不能改。任何截断都会让两侧分布不可比。
SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=-1)


# --- 扫描维度 --------------------------------------------------------------

MODELS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"]

# 桶长度: XLA 静态形状要求。同一条序列 pad 到不同桶 -> 不同编译图 -> 不同归约顺序
BUCKET_LENS = [512, 1024, 2048]

# 桶跨度: 实际长度 / 桶长度。H1b 说失配随跨度方差增长。
# 实现方式是筛选实际长度落在目标比例附近的序列, 而不是人为截断。
BUCKET_SPANS = [0.5, 0.75, 0.95]

# LNC (Logical NeuronCore Configuration): Trainium2 引入, 把多个物理 core
# 合并成一个逻辑 core。不同 LNC 下编译器的切分和归约决策不同。
# *** 这是 GPU 上没有对应概念的变量, 优先级高于 TP 扫描 ***
LNC_CONFIGS = [1, 2]

# 单芯片 8 core, TP 最多到 8。48xlarge 上可以扩到 [8, 16, 32]
TP_DEGREES = [1, 2, 4, 8]

BATCH_COMPOSITION = ["uniform", "mixed"]   # 同长 batch vs 混长 batch


# --- 优先级分组: 时间不够时按这个顺序砍 ------------------------------------

TIER_1 = "path"          # NEFF vs XLA 基础差值。没有它什么都不成立
TIER_2 = "span"          # 桶跨度扫描 (H1b)
TIER_3 = "lnc"           # LNC 配置
TIER_4 = "bucket,tp,model,composition"


@dataclass(frozen=True)
class RunConfig:
    model: str
    bucket_len: int
    bucket_span: float
    lnc: int
    tp: int
    composition: str = "uniform"

    @property
    def uid(self) -> str:
        """配置的短哈希, 用作文件名。改任何字段都会换 uid, 避免结果串味。"""
        s = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:10]

    def __str__(self):
        m = self.model.split("/")[-1]
        return f"{m}_b{self.bucket_len}_s{self.bucket_span}_lnc{self.lnc}_tp{self.tp}_{self.composition}"


def tier1_configs() -> list[RunConfig]:
    """最小可行实验: 一个配置, 只比 NEFF vs XLA。先跑通这个。"""
    return [RunConfig(MODELS[0], 1024, 0.75, lnc=2, tp=8)]


def tier2_configs() -> list[RunConfig]:
    """加桶跨度扫描。3 个配置。"""
    return [
        RunConfig(MODELS[0], 1024, span, lnc=2, tp=8) for span in BUCKET_SPANS
    ]


def tier3_configs() -> list[RunConfig]:
    """加 LNC。6 个配置。这是几天窗口内的推荐范围。"""
    return [
        RunConfig(MODELS[0], 1024, span, lnc=lnc, tp=8)
        for span in BUCKET_SPANS
        for lnc in LNC_CONFIGS
    ]


def full_configs() -> list[RunConfig]:
    """完整矩阵。144 个配置, 单芯片上跑不完, 列在这里供参考。"""
    return [
        RunConfig(m, b, s, lnc, tp, c)
        for m in MODELS
        for b in BUCKET_LENS
        for s in BUCKET_SPANS
        for lnc in LNC_CONFIGS
        for tp in TP_DEGREES
        for c in BATCH_COMPOSITION
    ]


# --- 路径 ------------------------------------------------------------------

DATA_DIR = "./data"
SAMPLE_DIR = "./results/samples"      # sample.py 的输出
RECOMPUTE_DIR = "./results/recompute" # recompute.py 的输出
ANALYSIS_DIR = "./results/analysis"   # analyze.py 的输出
