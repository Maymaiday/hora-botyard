# BotYard DexHand 最小资产包（给 Isaac Gym / 仿真用）

来源：`botyard_ws/src/ros2_botyard/dexhand_description`（v3.2 URDF + 网格）。
不含 Isaac Sim USD、ROS 控制、MoveIt。

## 目录

- `urdf/right_dexhand.urdf`、`urdf/left_dexhand.urdf`：网格路径已改成相对路径 `../meshes/...`，方便 Isaac Gym 直接加载
- `meshes/`：视觉/碰撞网格
- `ros_package/`：保留原始 `package://dexhand_description/...` 写法，仅作对照

## Isaac Gym 使用提示

1. 用绝对路径或相对工作目录的路径加载，例如：
   `urdf/right_dexhand.urdf`
2. 请确认 Gym 能解析 STL；必要时把碰撞/视觉网格路径再按你的工程结构调整。
3. 关节命名与 BotYard / Isaac Sim 场景中的 dexhand 一致（如 `FFJ3`、`THJ4` 等，具体以 URDF 为准）。
4. 本包只提供模型，不含在手操作任务代码；物体、控制、奖励需自行实现。

## 版本说明

- URDF 由 `dexhand.urdf.xacro` 生成的 v3.2 导出（`mjcf/v3_2/urdf/`）。
- 若与实物固件或其它仿真版本不一致，以你们当前硬件对应的 description 为准。
