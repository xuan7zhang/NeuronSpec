# Overnight summary

## stats.json 汇总

| run | n_tokens | abs_diff_mean | abs_diff_p50 | abs_diff_p99 | ratio_outside_tis_band | kl_k3 | seq_log_ratio_std | len_corr |
|---|---|---|---|---|---|---|---|---|
| b2048_lnc1 | 261839 | 0.01164 | 0.0001768 | 0.1328 | 0.001757 | 0.000492 | 0.7071 | 0.05769 |
| b2048_lnc2 | 261839 | 0.01164 | 0.0001813 | 0.1316 | 0.001913 | 0.0004909 | 0.6774 | 0.009225 |
| first | 261839 | 0.01165 | 0.0001804 | 0.1314 | 0.001814 | 0.0005237 | 0.6573 | 0.03248 |
| lnc1 | 261839 | 0.01164 | 0.0001746 | 0.1326 | 0.001753 | 0.0004865 | 0.7344 | 0.003266 |
| maxnew868_lnc2 | 443195 | 0.01121 | 0.0001137 | 0.1299 | 0.001706 | 0.000463 | 0.9111 | 0.002365 |
| tp1_lnc2 | 262074 | 0.008202 | 0.000132 | 0.1011 | 0.0009883 | 0.0002813 | 0.5252 | -0.04629 |
| tp2_lnc2 | 262105 | 0.01175 | 0.0001957 | 0.1324 | 0.001854 | 0.0004871 | 0.7113 | 0.04988 |
| tp8_lnc1 | 262144 | 0.01181 | 0.0002 | 0.1325 | 0.001949 | 0.00049 | 0.697 | nan |

## 采样文件: 截断率 / 长度 / 耗时

| sample | tp | lnc | max_new | rows | truncated | total_len p10/p50/p90/max | elapsed_s |
|---|---|---|---|---|---|---|---|
| maxnew868_tp4_lnc2.jsonl | 4 | 2 | 868 | 512 | 508 (99.2%) | 927/947/978/1024 | 1810 |
| neuron_Qwen3-1.7B_tp4_lnc1_b1024.jsonl | 4 | 1 | 512 | 512 | 510 (99.6%) | 571/591/622/668 | 1088 |
| neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl | 4 | 2 | 512 | 512 | 510 (99.6%) | 571/591/622/668 | 1112 |
| rep2_tp4_lnc2.jsonl | 4 | 2 | 512 | 512 | 510 (99.6%) | 571/591/622/668 | 1092 |
| tp1_lnc2.jsonl | 1 | 2 | 512 | 512 | 511 (99.8%) | 571/591/622/668 | 1466 |
| tp2_lnc2.jsonl | 2 | 2 | 512 | 512 | 511 (99.8%) | 571/591/622/668 | 1096 |
| tp8_lnc1.jsonl | 8 | 1 | 512 | 512 | 512 (100.0%) | 571/591/622/668 | 1068 |

## span 可行性 (bucket 1024, tol 0.08)

- maxnew868_tp4_lnc2: span 0.5 → 1 条, span 0.75 → 0 条, span 0.95 → 509 条
- neuron_Qwen3-1.7B_tp4_lnc1_b1024: span 0.5 → 0 条, span 0.75 → 0 条, span 0.95 → 0 条
- neuron_Qwen3-1.7B_tp4_lnc2_b1024: span 0.5 → 0 条, span 0.75 → 0 条, span 0.95 → 0 条
- rep2_tp4_lnc2: span 0.5 → 0 条, span 0.75 → 0 条, span 0.95 → 0 条
- tp1_lnc2: span 0.5 → 1 条, span 0.75 → 0 条, span 0.95 → 0 条
- tp2_lnc2: span 0.5 → 0 条, span 0.75 → 0 条, span 0.95 → 0 条
- tp8_lnc1: span 0.5 → 0 条, span 0.75 → 0 条, span 0.95 → 0 条

## 确定性 (同 seed 重跑)

- sample NEFF tp4/lnc2 run1 vs run2: token 序列相同 512/512; 序列且 logprob 逐位相同 512/512; 相同序列上 |Δlogp| mean 0.00e+00 max 0.00e+00
- recompute XLA lnc2 run1 vs run2 (同一输入): token 序列相同 512/512; 序列且 logprob 逐位相同 512/512; 相同序列上 |Δlogp| mean 0.00e+00 max 0.00e+00

## 跨配置: 同 seed 下采样轨迹是否相同

- tp4: lnc2 vs lnc1: token 序列相同 23/512; 序列且 logprob 逐位相同 0/512; 相同序列上 |Δlogp| mean 1.02e-02 max 5.61e-01
- lnc2: tp4 vs tp1: token 序列相同 13/512; 序列且 logprob 逐位相同 0/512; 相同序列上 |Δlogp| mean 9.32e-03 max 5.68e-01
- lnc2: tp4 vs tp2: token 序列相同 10/512; 序列且 logprob 逐位相同 0/512; 相同序列上 |Δlogp| mean 9.76e-03 max 2.82e-01
- lnc1: tp4 vs tp8: token 序列相同 8/512; 序列且 logprob 逐位相同 0/512; 相同序列上 |Δlogp| mean 1.02e-02 max 2.84e-01

## 各步骤耗时 (logs)

- analyze_b2048_lnc1.log: rc=0, wall 1s, 编译 0 张图
- analyze_b2048_lnc2.log: rc=0, wall 1s, 编译 0 张图
- analyze_lnc1.log: rc=0, wall 1s, 编译 0 张图
- analyze_maxnew868_lnc2.log: rc=0, wall 1s, 编译 0 张图
- analyze_tp1_lnc2.log: rc=0, wall 1s, 编译 0 张图
- analyze_tp2_lnc2.log: rc=0, wall 1s, 编译 0 张图
- analyze_tp8_lnc1.log: rc=0, wall 1s, 编译 0 张图
- recompute_b1024_tp1_lnc1.log: rc=0, wall 359s, 编译 2 张图
- recompute_b1024_tp1_lnc2.log: rc=0, wall 79s, 编译 0 张图
- recompute_rc_lnc1_b2048.log: rc=0, wall 269s, 编译 2 张图
- recompute_rc_lnc2_b2048.log: rc=0, wall 194s, 编译 2 张图
- recompute_rc_maxnew868_lnc2.log: rc=0, wall 295s, 编译 1 张图
- recompute_rc_tp1_lnc2.log: rc=0, wall 291s, 编译 2 张图
- recompute_rc_tp2_lnc2.log: rc=0, wall 78s, 编译 0 张图
- recompute_rc_tp8_lnc1.log: rc=0, wall 109s, 编译 0 张图
- recompute_rep2_rc_lnc2_b1024.log: rc=0, wall 75s, 编译 0 张图
- sample_maxnew868_tp4_lnc2.log: rc=0, wall 1944s, 编译 0 张图
- sample_rep2_tp4_lnc2.log: rc=0, wall 1228s, 编译 0 张图
- sample_tp1_lnc2.log: rc=0, wall 2049s, 编译 8 张图
- sample_tp2_lnc2.log: rc=0, wall 1492s, 编译 8 张图
- sample_tp4_lnc1.log: rc=0, wall 1355s, 编译 8 张图
- sample_tp4_lnc2.log: rc=0, wall 1397s, 编译 8 张图
- sample_tp8_lnc1.log: rc=0, wall 1290s, 编译 8 张图
