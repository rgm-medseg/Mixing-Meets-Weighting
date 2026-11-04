# 统一A2统计

DONE。直接复用已验证原概率/质量CSV，没有额外GPU前向。验证210张切片，GT-present有效205；全部210固定三前景macro和5张无前景切片的逐类FP同时保留。两种预测处理、训练标注子集分别保存，best/last不合并。患者同一有效切片配对，先切片等权再帧等权，共10患者。

|method_id|checkpoint_tag|signal|level|n|spearman|
|---|---|---|---|---|---|
|baseline|best|mean_max_probability|slice|205|-0.228286|
|baseline|best|mean_max_probability|patient|10|0.127273|
|baseline|best|q_B_auxiliary|slice|205|0.803302|
|baseline|best|q_B_auxiliary|patient|10|0.830303|
|baseline|last|mean_max_probability|slice|205|-0.243914|
|baseline|last|mean_max_probability|patient|10|0.115152|
|baseline|last|q_B_auxiliary|slice|205|0.809102|
|baseline|last|q_B_auxiliary|patient|10|0.818182|
|Full_main|best|mean_max_probability|slice|205|-0.217420|
|Full_main|best|mean_max_probability|patient|10|0.127273|
|Full_main|best|q_B_auxiliary|slice|205|0.807517|
|Full_main|best|q_B_auxiliary|patient|10|0.830303|
|Full_main|last|mean_max_probability|slice|205|-0.302852|
|Full_main|last|mean_max_probability|patient|10|0.090909|
|Full_main|last|q_B_auxiliary|slice|205|0.789474|
|Full_main|last|q_B_auxiliary|patient|10|0.854545|
|R1_v2_B|best|mean_max_probability|slice|205|-0.246641|
|R1_v2_B|best|mean_max_probability|patient|10|0.090909|
|R1_v2_B|best|q_B_auxiliary|slice|205|0.799174|
|R1_v2_B|best|q_B_auxiliary|patient|10|0.769697|
|R1_v2_B|last|mean_max_probability|slice|205|-0.259427|
|R1_v2_B|last|mean_max_probability|patient|10|0.139394|
|R1_v2_B|last|q_B_auxiliary|slice|205|0.797363|
|R1_v2_B|last|q_B_auxiliary|patient|10|0.854545|
|R2_fixed92|best|mean_max_probability|slice|205|-0.330053|
|R2_fixed92|best|mean_max_probability|patient|10|-0.175758|
|R2_fixed92|best|q_B_auxiliary|slice|205|0.795029|
|R2_fixed92|best|q_B_auxiliary|patient|10|0.806061|
|R2_fixed92|last|mean_max_probability|slice|205|-0.280540|
|R2_fixed92|last|mean_max_probability|patient|10|-0.078788|
|R2_fixed92|last|q_B_auxiliary|slice|205|0.797834|
|R2_fixed92|last|q_B_auxiliary|patient|10|0.830303|

统一前景等权间隔在baseline也呈强正相关，描述信号与冻结教师质量的条件关系。Full/baseline该量为离线辅助；R1是单切片控制代理，R2记录该信号但固定几何。Full真实论文版本映射NEEDS_MAPPING，不能把q_B相关结果改写成Full历史使用该控制量。无显著性检验、无因果训练收益结论。
