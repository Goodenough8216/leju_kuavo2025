#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from kuavo_msgs.msg import robotHandPosition

class SliderPublisher(QWidget):
    def __init__(self):
        super(SliderPublisher, self).__init__()

        self.initUI()
        self.pub = rospy.Publisher('control_robot_hand_position', robotHandPosition, queue_size=1)
        rospy.init_node('slider_publisher', anonymous=True)
        self.timer = QTimer()
        self.timer.timeout.connect(self.publish_positions)
        self.timer.start(100)  # 10 Hz

    def initUI(self):
        self.layout = QVBoxLayout()

        self.sliders = []
        self.labels = []

        # names = [
        #     'l_thumb_proximal_yaw', 'l_thumb_distal_pitch', 'l_index_proximal_finger', 'l_index_distal_finger',
        #     'l_middle_proximal_finger', 'l_middle_distal_finger', 'l_ring_proximal_finger', 'l_ring_distal_finger',
        #     'l_pinky_proximal_finger', 'l_pinky_distal_finger', 'r_thumb_proximal_yaw', 'r_thumb_distal_pitch',
        #     'r_index_proximal_finger', 'r_index_distal_finger', 'r_middle_proximal_finger', 'r_middle_distal_finger',
        #     'r_ring_proximal_finger', 'r_ring_distal_finger', 'r_pinky_proximal_finger', 'r_pinky_distal_finger'
        # ]
        names = [
            'l_thumb_proximal_yaw', 'l_thumb_distal_pitch', 'l_index_finger',
            'l_middle_finger', 'l_ring_finger', 'l_pinky_finger',
            'r_thumb_proximal_yaw', 'r_thumb_distal_pitch', 'r_index_finger',
            'r_middle_finger', 'r_ring_finger', 'r_pinky_finger'
        ]

        for name in names:
            label = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(0)
            self.labels.append(label)
            self.sliders.append(slider)
            self.layout.addWidget(label)
            self.layout.addWidget(slider)

        self.setLayout(self.layout)
        self.setWindowTitle('Hand Position Controller')
        self.show()


    def publish_positions(self):
        positions = [slider.value() for slider in self.sliders]
        # msg = Float64MultiArray(data=positions)

        msg = robotHandPosition()
        msg.header.stamp = rospy.Time.now()

        msg.left_hand_position = positions[0:6]
        msg.right_hand_position = positions[6:]

        self.pub.publish(msg)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = SliderPublisher()
    sys.exit(app.exec_())
