# 实现核查

旧Mask Token对应TRC（4个16维共享学习参考，没有预设解剖语义），旧AdaMix BCP-lite对应本文TCSMix；外部Senyh/AdaMix单独命名official AdaMix-CT。旧ContourReliabilityWeight/CRW对应CTW。

ACDC两视图共用空间rot/flip/rotate及resize，strong再cutout和ColorJitter。cutout人工标签为0；teacher目标由strong输入重新预测并每前景LCC，未强制cutout区域teacher目标为背景；CPS在全mixed输出上计算，CTW全部mixed像素参与，没有独立cutout遮挡排除。本轮保持历史行为，不凭正文疑问宣称实现错误。ICT强视图teacher概率同样从扰动输入取得。

CTW q来自两学生detached最大类别概率的均值。上下/左右后向绝对差相加，边界方向首项为0；逐图minmax分母下限1e-6；w=.25+r；类别均值L1(p1,detach p2)，全图均值，lambda.05，仅student1受此项梯度。E1每图mean(w)广播、均值和w detached，实际参与区域全mixed网格。E2/E3同一97²矩形、同位置随机规则/两视图共享掩码、互补交换和目标处理；诊断统计不修改训练损失。EMA系数.99包含router参数和state buffers。

E4只选宽18的普通1x1(16→18)/LN/GELU/1x1(18→16)零gamma残差，无token/原型/注意力。647参数vs641（+0.936%），接口与原特征16×256²一致；计算MAC/pixel投影576vs512，TRC另有两次tokeneinsum128，因此所计MAC合计576vs640（−10%），不含归一化/激活等。学生2与EMA教师都替换。宽度按最近参数数目决定，没有多宽度训练或测试选择。

COVID克隆输入后collate不再增强，两条mixed输入用同一mask，实际FFT交换net_input_1/2的低频空间分量（ifft取abs），不是随意声称幅度谱交换。纯复制情况下两输入的低频交换无额外内容扰动（可能仅浮点舍入）；模型dropout和两学生参数仍可产生不同预测。Full与baseline均由同一covid19数据适配读取manifest及collate；需用历史配置核对其他default-off扩展，不能把代码中未启用的25D功能算入COVID机制。

保存学生1/2、teacher、优化器、step、epoch、主进程Python/NumPy/torch/CUDA RNG、采样batch IDs和LR状态。FP32无AMP，amp_state=None。断点原子写盘，last为完整续训状态，best仅按验证提高选择；16worker增强RNG/预取队列无法完整恢复，故resume_exact=false。记录这一限制，不声称原历史恢复实现已位级精确。工程修改见source补丁及每run的revision_effective_source.py；旧代码未修改。


## 已证实的COVID缩放spacing问题

代码与原NIfTI头共同证实：512→256的in-plane resize未更新spacing；最终evaluator沿axis=2正确重建XYZ，所以轴序本身正确。此前仅根据dataset ZXY堆叠作出的轴序判断已撤回，以完整重建链路为准。历史距离调用的mm标签没有反映resize后的正确像素间距；前述单位说明只描述标签/调用一致性。没有修改旧值，另存corrected-derived-center-grid及corrected-native-grid评价。具体患者shape/spacing/原GT文件SHA见reports/covid_geometry_audit.csv，全部三口径原始统计见covid_existing_prediction_reevaluation.csv。新native口径对预测nearest回原canonical网格，GT不缩放，spacing(Z,X,Y)。该问题不改变ACDC grid距离对照，也不新增COVID训练。
