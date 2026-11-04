# 针对性方法来源

只核查以下两项，不开展综述，也不把它们的结果归给本项目设计。

- Adaptive Mix：使用预测与伪标签的代理分割损失和age控制规则/patch数量，并按confidence排序patch；本项目不移植其选择方向或多patch结构。[作者全文v2](https://arxiv.org/html/2407.21586v2)、[作者仓库](https://github.com/Senyh/AdaMix)。仓库及Crossref正式记录为Medical Image Analysis 108，103857，2026年2月；[正式DOI](https://doi.org/10.1016/j.media.2025.103857)。arXiv v2是2025-04-20，不能称作2026新预印本。正式元数据保存于reports/AdaMix_publication_metadata.json；本地原官方commit85361fb…仅作代码参照，未启动AdaMix训练。
- UCAD：用teacher entropy估计区域不确定性，在superpixel上做选择/位移，并有不确定性加权loss；本轮借鉴“控制对象需明确”的比较，不引入superpixel、位置选择或新loss。[作者全文v1](https://arxiv.org/html/2601.17366v1)、[作者代码](https://github.com/dcb937/UCAD)。作者[arXiv记录](https://arxiv.org/abs/2601.17366)声明Accepted by ISBI2026；本次实际核验的是2026-01-24预印本v1和作者代码，未核对最终会议出版页/DOI。代码树SHA及entropy/softmax位置存于reports/UCAD_source_tree_metadata.json、UCAD_code_inspection.json。

前景类别平衡概率/间隔及empty fallback0是本项目本轮候选设计。引用上述方法不构成该候选有效性证明。
