C1预检PASS。协议、数据/预训练/父运行身份、控制器、真实函数绑定、合成权重/梯度、隔离真实短程检查均通过。

短程共2个物理联合更新：首个工程检查在EMA整数状态断言发生类型错误，已保留日志和1步物理调用；修复检查后仅用剩余1步完成两学生/教师变化、EMA代数、原20帧验证、安全完整保存及独立加载。随后通过真实恢复入口零更新探测完整加载两学生/教师/优化器/RNG/采样器，模型状态逐tensor一致；未再做优化器调用。isolated_control_test另验证控制器stop标志、独占锁和旧PID身份拒绝。正式训练另建模型/优化器/seed42 RNG/采样器，从原预训练第0步开始。16 worker随机流/prefetch不能位级恢复。

唯一科学干预CTW逐图自身整图同均值均匀权重。概率L1、student2 stop-gradient、lambda0.05、TRC与TCSMix v2_B均保持。新增统计不读取GT、不增加前向/更新。旧Full仍NEEDS_MAPPING；历史预训练生成命令/训练H5逐文件历史哈希NA。当前资产已完整冻结。

正式第一次优化器调用前另存INITIALIZATION_ACTUAL.json；当前不宣称已发生。

启动后核验更新（2026-10-08T00:58:48.299012+08:00）：正式INITIALIZATION_ACTUAL.json已生成于第0步，两学生及教师共同骨干严格等于原预训练，TRC gamma=0；两优化器预训练状态逐tensor一致。实际FP32后端cudnn deterministic=True/benchmark=False/cudnn TF32=True/matmul TF32=False。首个200步完整检查点独立CPU加载PASS；ACTUAL_EFFECTIVE_BINDING.json证明真实编译入口与预检生成源码哈希一致。tmux PTY attach/detach后step130→184、日志23991→33495字节。正式训练继续中。
