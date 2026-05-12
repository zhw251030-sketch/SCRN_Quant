# 本地脚本目录

本目录用于放置 `ch4_2_exp01_w4a32_visual_recovery` 实验的本地图件生成脚本。

按照当前 git 原则，脚本文件作为本地工具使用，不纳入 git 跟踪。git 只提交结果图、manifest、选择说明、README 和日志。

脚本使用要求：

- 尽量从 `../experiment_info.json` 读取实验路径。
- 候选图只能写入 `../candidates/`。
- 每个候选图版本必须写入一个对应的 `manifest_vXXX.json`。
- 不覆盖已有的 `vXXX` 图件或 manifest。
- 生成细节必须记录到 `../../../ARTIFACTS_LOG.md`。
