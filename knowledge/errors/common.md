# 常见问题

- `Patch size exceeds output extent`：模型 Patch 大小超过当前实验模式的输出网格，应减小 Patch 或扩大源裁剪范围。
- `No crop manifest is available`：TIFF 融合前需要先检查原始 TIFF 并执行裁剪，或直接提供已有 crop manifest。
- `No runtime preflight result`：先执行环境预检，或仅在明确排查时使用跳过预检选项。
- `strict_metadata` 下出现 CRS、Bounds 或分辨率不一致：若数据已在系统外完成配准且模型采用显式像素裁剪，可改用 `external_registration`；该选择只会降级元数据检查，不会自动修正配准误差。
