# first: Qwen3-1.7B, trn2.3xlarge, NEFF(tp4,lnc2) vs XLA(tp1,lnc2), bucket 1024

日期: 2026-09-08. 数据: data/prompts.jsonl (512 条 GSM8K test, checksum fb8e2ab826e12dfe)

## 两侧配置
| | sample (推理侧) | recompute (训练侧) |
|---|---|---|
| 栈 | vLLM 0.11.0 + vllm-neuron 0.2.2+lts + NxDI 0.7.15063 (NEFF) | HF transformers 4.56.2 + torch_xla lazy tensor (XLA 图) |
| venv | /opt/aws_neuronx_venv_pytorch_inference_vllm | 同一个 venv (torch 2.8.0 / torch_xla 2.8.1) |
| TP / LNC / cores | tp=4, LNC=2, NEURON_RT_VISIBLE_CORES=0-3 | tp=1, LNC=2, core 0 |
| batch | max_num_seqs=32 | --batch-size 1 |
| 采样 | temperature=1.0 top_p=1.0 top_k=-1, on_device_sampling_config=None (CPU 全词表采样) | n/a |
| logprob 计算 | NEFF 输出完整 logits -> vLLM Sampler fp32 log_softmax | XLA 前向 bf16 logits -> .float() log_softmax (图内) |
| 编译缓存 | /home/ec2-user/neuron-cache/infer (8 NEFF, 70 MB) | /home/ec2-user/neuron-cache/train (14 NEFF, 113 MB) |
| 冷编译 | ~3.5 min (8 张图, CTE -O1 / TKG -O2) | ~3.8 min (B=1,T=1024 单张前向图) |
| 运行 | 1111.9 s (CPU 采样瓶颈) | 62.9 s |

neuronx-cc 2.22.12471.0, runtime 2.29.40.0, driver 2.25.4.0.

## 注意事项
- 截断率 510/512 (finish_reason=length): Qwen3-1.7B 对裸 completion prompt 不发 EOS。
  几乎所有序列长度都是 512, 所以 len_corr 和 fig4 在这组数据上没有信息量。
- 训练侧 tp=1 而推理侧 tp=4: TP 是混杂因素, 现有 recompute.py 是普通 HF 模型无法做 TP>1。
- 训练侧 batch 必须为 1: B>=2 时 neuronx-cc 对 [B*T, 151936] 的 lm_head+log_softmax 估算 HBM 56 GB (>24 GB/逻辑核), 与 padding 无关。
- 采样侧 13% 的 logprob 是精确 0.0 (fp32 log_softmax 在 gap 极大时舍入到 0), 训练侧同位置为 ~-1e-9, 差异量级 1e-9, 对指标无影响。
- 三个 torch 2.9 的 venv 在本 AMI (glibc 2.34) 上 torch_xla 需要 GLIBC_2.35, 无法 import; 训练侧只能用 vLLM venv 的 torch 2.8 栈。
