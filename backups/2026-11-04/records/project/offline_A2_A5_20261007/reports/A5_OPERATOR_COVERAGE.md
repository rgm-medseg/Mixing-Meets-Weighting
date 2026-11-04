# 算子覆盖

输入1×1×256×256，FP32 eval。torch.profiler(with_flops=True,record_shapes=True)原生事件保存在A5_operators_*.csv。PyTorch版本见环境。模块hooks统计Conv2d/Linear：输出元素数×输入通道(分组修正)×卷积核面积；偏置加法另列。einsum的原生bmm计数逐项核验，人工公式仅补至模块MAC表，未再将原生profile总数相加。统一1 MAC=2 FLOPs。

TRC project/inverse_project由Conv hooks覆盖，各16×16×256² MAC；两次einsum各256²×4×16 MAC；softmax、logit尺度、残差融合、LayerNorm的算式与数值另见A5_TRC_analytic_operations.csv。LayerNorm公式按一次共享中心化、均值/方差及仿射的7D+2标量操作约定；softmax减max、exp、求和、除法为4T−1，加上max比较另不计。这是明确的标量约定，exp/sqrt各记1不代表硬件cycle。

原生profile未完整报告BatchNorm、LayerNorm、ReLU、MaxPool比较、双线性上采样、softmax、exp/log/norm、isfinite/比较/规约以及复制/CPU同步的运算成本；argmax及原forward遥测也保留。完整推理FLOPs=NA/PARTIAL，主表提供已覆盖MAC及其2倍FLOPs，不将遗漏静默记0。辅助TRC标量公式不会将主表伪装为完整计数。训练FFT、全部损失、反向、EMA运算未完整计数，训练总FLOPs=NA。

TCSMix/CTW在混合、损失阶段执行，真实student1部署结构只有UNet或UNet+TRC；训练系统参数为两学生加冻结EMA教师，单模型计时不包含第二学生/教师。
