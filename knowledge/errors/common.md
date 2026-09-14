# 常见问题

- `Patch size exceeds output extent`：模型 Patch 大小超过当前实验模式的输出网格，应减小 Patch 或扩大源裁剪范围。
- `No crop manifest is available`：TIFF 融合前需要先检查原始 TIFF 并执行裁剪，或直接提供已有 crop manifest。
- `No runtime preflight result`：先执行环境预检，或仅在明确排查时使用跳过预检选项。
