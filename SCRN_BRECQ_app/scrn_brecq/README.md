# SCRN BRECQ 应用目录

本目录用于存放将 BRECQ 后训练量化算法迁移应用到 SCRN 模型的独立实现代码。

## 设计原则

- 只参考 `BRECQ-main/` 中的算法思想和实现结构，不直接导入其中的源码模块。
- 优先复用 `SCRN_BRECQ_app/scrn_repro/` 中已经完成的 SCRN 复现模型、数据读取和评估逻辑。
- BRECQ 相关适配代码集中放在本目录，避免污染 `SCRN-main/`、`BRECQ-main/` 和 `scrn_repro/`。
- 运行产物、模型权重、缓存、日志和数据文件不提交到 Git。

## 初始目录结构

```text
scrn_brecq/
├── README.md
├── DEVELOPMENT_LOG.md
├── __init__.py
├── configs/
│   └── default_quant_config.json
├── cli/
│   └── __init__.py
├── data/
│   └── __init__.py
├── model/
│   └── __init__.py
├── quant/
│   └── __init__.py
├── utils/
│   └── __init__.py
└── runs/
    └── README.md
```

后续会按小目标逐步补齐：

1. SCRN checkpoint 加载适配。
2. calibration 数据加载器。
3. BRECQ 基础量化层。
4. AdaRound 权重量化重构。
5. SCRN block/layer 重构流程。
6. 量化模型评估脚本。

