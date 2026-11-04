# TCSMix 控制修订证据

先读TCSMIX_CONTROL_DESIGN_CN.md、SHORT_RUN_REPORT_CN.md、FULL_READINESS_CN.md、SCOPE_DIFF_AUDIT.md、EDITOR_HANDOFF_CN.md。正式实验未授权，当前WAITING_FOR_FULL_APPROVAL，原队列保持暂停。

control_diagnostics.csv：同输入/同checkpoint的old/A/B控制与原LCC支持，scope=batch与single_sample_response分开；row cohort不同勿当训练总体分布。short_trace.csv：唯一B短程真实1–240步、两optimizer调用总480；old_q/q_A是同v2 teacher的回放，不是独立old/A训练。
analysis/包含尺寸分布、相关性、GT类别存在性、概率受控检查、微基准成本、PNG/SVG图与来源。reports/offline_A_initial保留先A诊断及原脚本；report/control_diagnostics_before_GT_metadata保留追加GT语义字段前的同分数版本，不能计为独立样本/训练。
checkpoint路径与hash见reports/short_checkpoint_checks.json、offline_checkpoint_inventory.json，weights只在服务器，不入包。无原始影像/GT数组/凭据。

可审查入口：scripts/offline_diagnostics.py --compare-b（已有结果不能当额外独立实验），check_controller.py、preflight_control.py、run_short_check.py、postcheck_control.py。短程现已完成，run_short_check/入口的复跑会拒绝现有run或预算；不要修改配置绕过。正式R1/R2 planned配置会在初始化前被当前入口拒绝，待用户正式确认后另立阶段和服务器队列。lock_R2_from_R1.py只在R1完整真实轨迹可用后产出新锁定配置。
源代码与变更patch、实际命令/日志、环境、input SHA、五轮检查和静态SHA256清单均入包。原结果和被暂停E1不会被本阶段脚本恢复。
