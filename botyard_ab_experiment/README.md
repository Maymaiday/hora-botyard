# BotYard A/B 续训补丁

A 为原奖励续训；B 仅在原奖励上增加以倾斜角减小量计算的恢复 shaping，不增加动作/观测维度，也**不实现显式换指状态机**。两组均从 1M best checkpoint 的相同网络与优化器出发；B 因奖励定义改变，重置保存 Best 的比较基准，但不重置模型和优化器。

必须从 hora 项目根目录运行 apply_ab_patch.py。run_ab.sh 的 6M 指**累计**环境步数（续训约 5M）。脚本假设项目在 /home/grelcy/hora；请核实目录和 1M best.pth 存在。

B 的 abRecovery.weight 默认 0.10，属于实验超参数，尚未经 Isaac Gym 验证。该 shaping 只在倾斜减小的步骤奖励，倾斜增加时扣分，不保证解决换指操作，也不保证仿真稳定。

评价请使用相同 size=1.0 的种子集合，重点观察稳定时间、累计净旋转角、倾斜和失败率。不同奖励函数下，A/B 的 raw reward 数值不能直接横向比较。
