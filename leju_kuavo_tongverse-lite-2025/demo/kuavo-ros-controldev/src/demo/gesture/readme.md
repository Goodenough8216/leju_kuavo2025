# Demo - Execute Gesture

## 描述

调用手势服务接口执行对应名称的预设手势, 手势列表可通过`gesture/list`服务接口获取, 或者查阅[接口文档](../../../docs/运动控制API.md#主要topics和srv)
## 示例

### 手势列表
获取所有预设的手势列表.
```bash
python3 list_all_gestures.py
```

### ok 手势
执行 ok 手势.
```bash
python3 gesture_client.py ok
```

### 数字 1~8手势
```bash
python3 number_gesture.py
```