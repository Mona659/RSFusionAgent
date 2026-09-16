# 输入数据契约 / Input Data Contract

关键词：真实实验、模拟实验、输入数据要求、T1 MS、T1 HS、T2 MS、T2 HS、real experiment、simulation experiment、input requirements、pseudo reference。

## 真实实验 / Real experiment

- 推理必需三景输入：辅助时相多光谱 **T1 MS**、辅助时相高光谱 **T1 HS**、目标时相多光谱 **T2 MS**。
- 真实实验没有原生的目标时相高光谱真值 **T2 HS**；因此不能将真实实验指标解释为与原生真值的直接比较。
- 可以额外提供 T2 HS 作为可视化或评估用伪标签。系统仅将其插值到 MS 网格，结果必须标注为 **伪参考评估 / pseudo-reference evaluation**。
- 原始 TIFF 工作流先检查输入，再使用显式 MS/HS 3×空间对应关系裁剪；系统不做自动配准、重投影或重采样。
- 默认“已外部配准”声明适用于沿用原模型数据流程的输入：波段数、3×像素尺寸关系和裁剪窗口范围仍是硬约束；轻微 CRS、Bounds 或分辨率元数据差异仅记录为警告。它不证明残余配准误差足够小，使用者需确认外部配准质量。
- “严格元数据一致校验”才要求 CRS、Bounds、分辨率完全满足容差，用于必须强制空间元数据一致的实验。

## 模拟实验 / Simulation experiment

- 同样需要 T1 MS、T1 HS、T2 MS；此外必须提供目标时相原始 **T2 HS**。
- T2 HS 原始裁剪用作低分辨率真值；模型输入依据 3×空间尺度关系进行模糊和下采样，随后可计算 PSNR、SAM、SSIM 等全参考指标。

## H5 演示模式 / H5 evaluation mode

- 直接使用已预处理的辅助时相与目标时相 H5 文件中的 `test` Patch。
- H5 输入已经遵守既有模型数据契约，不重复执行 TIFF 裁剪和下采样。
