# EdgeArm：单腕部视觉推块

## 核心方法

最终方法按四阶段组织：连续序列学习空间记忆；DAgger采集学生偏离状态并训练动作；
把教师标签投影到相机/工作区/关节联合约束；再用访问状态图像适配关键点定位。
各阶段冻结其他模块，用旧数据功能回练抑制遗忘，以完整闭环表现选型。
历史ACT、在线残差RL和多来源采集代码保留，但不将最终方案误称为ACT或RL模型。

冻结候选在8个新场景组、九路线等权的72次测试中成功51次（70.83%）。
成功必须在目标区域连续稳定保持3秒。部署只用腕部RGB、报告关节、动作历史及标定。
当前结果是名义仿真、任务条件化起点，含固定观察和收尾程序，不代表实机验证。

## 使用

四个最终权重也可直接从 [GitHub Release v0.1.0](https://github.com/Tangtaizong-BUAA/EdgeArm/releases/tag/v0.1.0)
下载，附有 `checksums.json`。将四个 `.pt` 文件放入 `checkpoints/run102` 即可用于评估。
它们与 Hugging Face 发布版本完全一致，采用 Apache-2.0；不包含训练数据。

详见英文首页的安装命令；必须保留源码检出目录中的SO101资源。

```bash
edgearm doctor
edgearm download --output checkpoints/run102
edgearm evaluate --weights checkpoints/run102 --output outputs/replay --workers 4
edgearm stages
edgearm run recipes/spatial_memory.json
```

`run`默认只展示命令，检查本地数据和模型路径后加`--execute`执行。
`evaluate`默认复现已公开的历史种子，不冒充新的独立泛化测量。

入口索引：

- [完整训练主线](docs/training.md)：训练谁、冻结谁、输入输出、采样和损失。
- [数据协议](docs/data.md)：RGB/本体历史与特权标签隔离、划分与反例用途。
- [模型卡](docs/model-card.md)：权重组件、真实结果、使用范围。
- [复现与发布状态](docs/reproducibility.md)：测试与未完成项分开报告。

代码和发布权重采用Apache-2.0，第三方资产保留原始归属。
