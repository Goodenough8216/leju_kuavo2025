### 代码说明

#### 功能
该代码实现了一个 ROS 节点，用于订阅图像话题并显示接收到的图像。它使用 OpenCV 将 ROS 图像消息转换为 OpenCV 格式，并在窗口中实时显示。

#### 代码参数
- **msg (sensor_msgs/Image)**: ROS 图像消息，包含图像数据和相关元信息。

#### 逻辑
1. **导入库**:
   - 导入 `rospy` 库以使用 ROS 的 Python 接口。
   - 导入 `Image` 消息类型，用于处理图像数据。
   - 导入 `CvBridge` 和 `CvBridgeError`，用于在 ROS 图像消息和 OpenCV 图像之间进行转换。
   - 导入 `cv2` 库，用于图像处理和显示。

2. **定义回调函数**:
   - `image_callback(msg)`:
     - 创建 `CvBridge` 对象，用于图像格式转换。
     - 尝试将 ROS 图像消息转换为 OpenCV 格式的图像（BGR 格式）。
     - 如果转换失败，记录错误信息并返回。
     - 使用 OpenCV 显示图像，窗口名称为 "Camera Image"。
     - 等待 1 毫秒以处理窗口事件。

3. **主程序**:
   - `main()`:
     - 初始化 ROS 节点，节点名称为 `image_subscriber`。
     - 订阅图像话题 `/camera/color/image_raw`，当接收到消息时调用 `image_callback` 函数。
     - 使用 `rospy.spin()` 保持节点运行，直到节点被关闭。

4. **入口点**:
   - 如果此脚本是主程序，则调用 `main()` 函数。

### 总结
该代码提供了一个简单的 ROS 图像订阅者，能够实时接收和显示来自相机的图像数据，适用于机器人视觉处理和监控应用。

#### 相机常用Topic
深度相机
* /camera/depth/image_rect_raw # 深度相机图像
* /camera/depth/camera_info    # 深度相机内参（由realsense出厂前设置标定好）
* /camera/depth/color/points   # 带有深度的RGB点云（Point2D）

RGB相机
* /camera/color/image_raw    # RGB相机图像
* /camera/color/camera_info  # RGB相机内参（由realsense出厂前设置标定好）

深度对齐RGB相机
* /camera/aligned_depth_to_color/image_raw   # 深度对齐RGB后的深度图像
* /camera/aligned_depth_to_color/camera_info # 深度对齐RGB后的相机内参