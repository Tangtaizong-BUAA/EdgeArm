# SO101 机器人 - URDF 与 MuJoCo 描述

本仓库包含 SO101 机器人的 URDF 和 MuJoCo（MJCF）文件。

## 概览

- 机器人模型文件由 [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) 插件从 Onshape 中设计的 CAD 模型生成。
- 生成后的 URDF 已被修改，以允许 mesh 使用相对路径，而不是 `package://...`。
- 由于基础部分的碰撞 mesh 在仿真和规划中表现有问题，因此已经移除。

## 校准方法

MuJoCo 文件 `scene.xml` 支持两种不同校准方式的 SO101 机器人文件：

- **新校准（默认）**：每个关节的虚拟零位设置在该关节运动范围的**中间**。使用 `so101_new_calib.xml`。
- **旧校准**：每个关节的虚拟零位设置为机器人**完全水平伸展**时的构型。使用 `so101_old_calib.xml`。

要在两种校准方法之间切换，请修改 `scene.xml` 中包含的机器人文件。

## 电机参数

机器人中使用的 STS3215 电机参数改编自 [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini)。

## 夹爪说明

在 LeRobot 中，夹爪被表示为一个**线性关节**，其中：

* `0` = 完全闭合
* `100` = 完全张开

这个映射**尚未反映**在当前的 URDF 和 MuJoCo 文件中。

---

欢迎创建 issue 或贡献改进！
