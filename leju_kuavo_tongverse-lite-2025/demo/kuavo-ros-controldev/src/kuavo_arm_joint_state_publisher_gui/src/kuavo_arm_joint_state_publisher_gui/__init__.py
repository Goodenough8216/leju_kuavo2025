# Software License Agreement (BSD License)
#
# Copyright (c) 2010, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
import random

import rospy

from python_qt_binding.QtCore import pyqtSlot
from python_qt_binding.QtCore import Qt
from python_qt_binding.QtCore import Signal
from python_qt_binding.QtCore import QTimer
from python_qt_binding.QtGui import QFont
from python_qt_binding.QtWidgets import QApplication
from python_qt_binding.QtWidgets import QHBoxLayout
from python_qt_binding.QtWidgets import QLabel
from python_qt_binding.QtWidgets import QLineEdit
from python_qt_binding.QtWidgets import QPushButton
from python_qt_binding.QtWidgets import QSlider
from python_qt_binding.QtWidgets import QVBoxLayout
from python_qt_binding.QtWidgets import QGridLayout
from python_qt_binding.QtWidgets import QScrollArea
from python_qt_binding.QtWidgets import QSpinBox
from python_qt_binding.QtWidgets import QWidget
from sensor_msgs.msg import JointState
from kuavo_msgs.srv import changeArmCtrlMode
from kuavo_msgs.msg import sensorsData

RANGE = 10000


class JointStatePublisherGui(QWidget):
    sliderUpdateTrigger = Signal()

    def __init__(self, title, jsp, num_rows=0):
        super(JointStatePublisherGui, self).__init__()
        self.setWindowTitle(title)
        self.jsp = jsp
        self.joint_map = {}
        self.vlayout = QVBoxLayout(self)
        self.scrollable = QWidget()
        self.gridlayout = QGridLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        
        # 保存最新的传感器数据
        self.latest_sensor_data = None
        
        # 添加控制启用标志，默认为禁用状态
        self.control_enabled = False
        
        # 创建发布器，发布关节数据到/kuavo_arm_traj
        self.arm_publisher = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)
        
        # 创建订阅器，订阅/sensors_data_raw
        self.sensor_subscriber = rospy.Subscriber('/sensors_data_raw', sensorsData, self.sensor_data_callback)
        
        # 创建定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.publish_arm_joints)
        self.timer.start(5)

        self.jsp.set_source_update_cb(self.source_update_cb)

        font = QFont("Helvetica", 9, QFont.Bold)

        ### Generate sliders ###
        sliders = []
        for name in self.jsp.joint_list:
            if name not in self.jsp.free_joints:
                continue
            joint = self.jsp.free_joints[name]

            if joint['min'] == joint['max']:
                continue
                
            # 只显示名称中包含"arm"的关节
            if "arm" not in name.lower():
                continue

            joint_layout = QVBoxLayout()
            row_layout = QHBoxLayout()

            label = QLabel(name)
            label.setFont(font)
            row_layout.addWidget(label)
            display = QLineEdit("0.00")
            display.setAlignment(Qt.AlignRight)
            display.setFont(font)
            display.setReadOnly(True)
            row_layout.addWidget(display)

            joint_layout.addLayout(row_layout)

            slider = QSlider(Qt.Horizontal)

            slider.setFont(font)
            slider.setRange(0, RANGE)
            slider.setValue(int(RANGE/2))

            joint_layout.addWidget(slider)

            self.joint_map[name] = {'slidervalue': 0, 'display': display,
                                    'slider': slider, 'joint': joint}
            # Connect to the signal provided by QSignal
            slider.valueChanged.connect(lambda event,name=name: self.onValueChangedOne(name))

            sliders.append(joint_layout)

        # Determine number of rows to be used in grid
        self.num_rows = num_rows
        # if desired num of rows wasn't set, default behaviour is a vertical layout
        if self.num_rows == 0:
            self.num_rows = len(sliders)  # equals VBoxLayout
        # Generate positions in grid and place sliders there
        self.positions = self.generate_grid_positions(len(sliders), self.num_rows)
        for item, pos in zip(sliders, self.positions):
            self.gridlayout.addLayout(item, *pos)

        # Set zero positions read from parameters
        self.center()

        # Synchronize slider and displayed value
        self.sliderUpdate(None)

        # Set up a signal for updating the sliders based on external joint info
        self.sliderUpdateTrigger.connect(self.updateSliders)

        self.scrollable.setLayout(self.gridlayout)
        self.scroll.setWidget(self.scrollable)
        self.vlayout.addWidget(self.scroll)

        # Buttons for controlling the arm joint state publisher
        self.enable_button = QPushButton('Enable Control', self)
        self.enable_button.clicked.connect(self.enable_control)
        self.vlayout.addWidget(self.enable_button)
        
        self.disable_button = QPushButton('Disable Control', self)
        self.disable_button.clicked.connect(self.disable_control)
        self.vlayout.addWidget(self.disable_button)
        
        # # 重新添加Center按钮
        # self.center_button = QPushButton('Center', self)
        # self.center_button.clicked.connect(self.center_event)
        # self.vlayout.addWidget(self.center_button)
        
        # 添加Refresh按钮，用于从传感器数据更新关节位置
        self.refresh_button = QPushButton('Refresh from Sensors', self)
        self.refresh_button.clicked.connect(self.refresh_from_sensors)
        self.vlayout.addWidget(self.refresh_button)
        
        self.maxrowsupdown = QSpinBox()
        self.maxrowsupdown.setMinimum(1)
        self.maxrowsupdown.setMaximum(len(sliders))
        self.maxrowsupdown.setValue(self.num_rows)
        self.maxrowsupdown.valueChanged.connect(self.reorggrid_event)
        self.vlayout.addWidget(self.maxrowsupdown)
        self.setLayout(self.vlayout)

    def source_update_cb(self):
        self.sliderUpdateTrigger.emit()

    def onValueChangedOne(self, name):
        # A slider value was changed, but we need to change the joint_info metadata.
        joint_info = self.joint_map[name]
        joint_info['slidervalue'] = joint_info['slider'].value()
        joint = joint_info['joint']
        joint['position'] = self.sliderToValue(joint_info['slidervalue'], joint)
        joint_info['display'].setText("%.3f" % joint['position'])

    @pyqtSlot()
    def updateSliders(self):
        self.update_sliders()

    def update_sliders(self):
        for name, joint_info in self.joint_map.items():
            joint = joint_info['joint']
            joint_info['slidervalue'] = self.valueToSlider(joint['position'],
                                                           joint)
            joint_info['slider'].setValue(joint_info['slidervalue'])

    def center_event(self, event):
        self.center()

    def center(self):
        rospy.loginfo("Centering")
        for name, joint_info in self.joint_map.items():
            joint = joint_info['joint']
            joint_info['slider'].setValue(self.valueToSlider(joint['zero'], joint))

    def reorggrid_event(self, event):
        self.reorganize_grid(event)

    def reorganize_grid(self, number_of_rows):
        self.num_rows = number_of_rows

        # Remove items from layout (won't destroy them!)
        items = []
        for pos in self.positions:
            item = self.gridlayout.itemAtPosition(*pos)
            items.append(item)
            self.gridlayout.removeItem(item)

        # Generate new positions for sliders and place them in their new spots
        self.positions = self.generate_grid_positions(len(items), self.num_rows)
        for item, pos in zip(items, self.positions):
            self.gridlayout.addLayout(item, *pos)

    def generate_grid_positions(self, num_items, num_rows):
        if num_rows == 0:
            return []
        positions = [(y, x) for x in range(int((math.ceil(float(num_items) / num_rows)))) for y in range(num_rows)]
        positions = positions[:num_items]
        return positions

    def randomize_event(self, event):
        self.randomize()

    def randomize(self):
        rospy.loginfo("Randomizing")
        for name, joint_info in self.joint_map.items():
            joint = joint_info['joint']
            joint_info['slider'].setValue(
                    self.valueToSlider(random.uniform(joint['min'], joint['max']), joint))

    def sliderUpdate(self, event):
        for name, joint_info in self.joint_map.items():
            joint_info['slidervalue'] = joint_info['slider'].value()
        self.update_sliders()

    def valueToSlider(self, value, joint):
        return int((value - joint['min']) * float(RANGE) / (joint['max'] - joint['min']))

    def sliderToValue(self, slider, joint):
        pctvalue = slider / float(RANGE)
        return joint['min'] + (joint['max']-joint['min']) * pctvalue

    def publish_arm_joints(self):
        """发布包含arm的关节信息到/kuavo_arm_traj话题"""
        # 只有在控制启用时才发布关节信息
        if not self.control_enabled:
            return
            
        arm_joints = JointState()
        arm_joints.header.stamp = rospy.Time.now()
        
        for name, joint_info in self.joint_map.items():
            if "arm" in name.lower():
                arm_joints.name.append(name)
                joint = joint_info['joint']
                # 使用角度值而不是弧度值
                arm_joints.position.append(math.degrees(joint['position']))
        
        # 只有当有arm关节时才发布
        if len(arm_joints.name) > 0:
            self.arm_publisher.publish(arm_joints)

    def disable_control(self):
        """调用服务设置控制模式为1（禁用控制）"""
        try:
            rospy.wait_for_service('/arm_traj_change_mode', timeout=1.0)
            change_mode = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            response = change_mode(1)
            # 设置控制标志为禁用
            self.control_enabled = False
            rospy.loginfo("Disabled control, service response: %s", response)
        except rospy.ROSException as e:
            rospy.logerr("Service call failed: %s", e)

    def enable_control(self):
        """调用服务设置控制模式为2（启用控制）"""
        try:
            rospy.wait_for_service('/arm_traj_change_mode', timeout=1.0)
            change_mode = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            response = change_mode(2)
            # 设置控制标志为启用
            self.control_enabled = True
            rospy.loginfo("Enabled control, service response: %s", response)
        except rospy.ROSException as e:
            rospy.logerr("Service call failed: %s", e)

    def sensor_data_callback(self, data):
        """传感器数据回调函数，保存最新的传感器数据"""
        self.latest_sensor_data = data
    
    def refresh_from_sensors(self):
        """从传感器数据更新关节位置"""
        if self.latest_sensor_data is None:
            rospy.logwarn("No sensor data received yet")
            return
            
        # 确保joint_q数组有足够的元素
        if len(self.latest_sensor_data.joint_data.joint_q) < 26:
            rospy.logwarn("Sensor data does not have enough joint_q values")
            return
            
        # 获取第13到26的joint_q值
        arm_joint_values = self.latest_sensor_data.joint_data.joint_q[12:26]
        
        # 将这些值应用到对应的arm关节
        arm_joint_names = [name for name in self.joint_map.keys() if "arm" in name.lower()]
        
        # 确保关节数量与获取的值数量匹配
        if len(arm_joint_names) != len(arm_joint_values):
            rospy.logwarn(f"Mismatch between arm joints ({len(arm_joint_names)}) and sensor values ({len(arm_joint_values)})")
            # 如果数量不匹配，尝试使用可用的最小数量
            min_len = min(len(arm_joint_names), len(arm_joint_values))
            arm_joint_names = arm_joint_names[:min_len]
            arm_joint_values = arm_joint_values[:min_len]
            
        # 更新关节位置
        for name, value in zip(arm_joint_names, arm_joint_values):
            joint_info = self.joint_map[name]
            joint = joint_info['joint']
            # 从传感器获取的可能是弧度值，但在界面上显示和使用的是度数
            # 所以这里不需要转换，直接设置即可
            radians_value = value  # 传感器数据中的值可能已经是弧度
            joint['position'] = radians_value
            joint_info['slidervalue'] = self.valueToSlider(radians_value, joint)
            joint_info['slider'].setValue(joint_info['slidervalue'])
            joint_info['display'].setText("%.3f" % radians_value)
            
        rospy.loginfo("Refreshed arm joint positions from sensor data")
