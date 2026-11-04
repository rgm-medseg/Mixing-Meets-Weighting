# 依赖准备

绝对解释器/home/renguanming/.conda/envs/beta_fft/bin/python，torch2.4.1+cu121，torchvision.19.1+cu121，h5py/MedPy/scikit-image/tensorboardX/matplotlib/SimpleITK已实测导入；CUDA可用。使用A10040GB，驱动550.54.14。

官方AdaMix固定版本已完整git clone在external/AdaMix，所有代码本地可用；新增easydict1.13/unfoldNd.2.3/pandas2.3.3/thop.1.1.post2209072238下载到独立external/python_deps，不修改原环境。pandas仅上游报告用，运行入口去除这一未使用import，不需要其pytz。官方开发环境Python3.8/torch1.11，实际采用当前2.4.1并用真实batch smoke检查兼容性，差异明示。

后台禁用在线跟踪登录；数据、初始化、源码、依赖在服务器持久磁盘，无临时授权链接、客户端挂载、代理隧道依赖。GPU资源不足时WAITING_RESOURCE每30秒检查，不停止其他任务，不OOM后静默改B/loss/budget，不自动重试代码错误。
