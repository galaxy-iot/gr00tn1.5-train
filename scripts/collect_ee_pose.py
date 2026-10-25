#!/usr/bin/env python3
"""Publish measured link/TCP poses from a complete, timestamped JointState."""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from urdf_fk import ForwardKinematics, multiply, origin, quaternion, vector


DEFAULT_URDF = '/home/debian/ros2/motion-core/src/episode1_urdf_1113/urdf/episode1_urdf_1113.urdf'


class PoseCollector(Node):
    def __init__(self):
        super().__init__('collect_ee_pose')
        defaults = {
            'urdf_path': DEFAULT_URDF, 'base_link': 'base_link', 'tip_link': 'link6',
            'joint_topic': '/joint_states', 'pose_topic': '/end_effector/pose',
            'tcp_frame': '', 'tcp_xyz': '0 0 0', 'tcp_rpy': '0 0 0',
            'tcp_pose_topic': '/end_effector/tcp_pose',
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        get = lambda key: self.get_parameter(key).value
        self.base, self.tip = get('base_link'), get('tip_link')
        self.fk = ForwardKinematics(get('urdf_path'), self.base, self.tip)
        self.publisher = self.create_publisher(PoseStamped, get('pose_topic'), 10)
        xyz, rpy = vector(get('tcp_xyz'), [0, 0, 0]), vector(get('tcp_rpy'), [0, 0, 0])
        self.tcp = origin(xyz, rpy)
        self.tcp_frame = get('tcp_frame')
        if not self.tcp_frame and any(v != 0 for v in xyz+rpy):
            raise ValueError('Set tcp_frame when specifying a TCP transform')
        self.tcp_publisher = None
        if self.tcp_frame:
            if get('pose_topic') == get('tcp_pose_topic'):
                raise ValueError('Link and TCP pose topics must differ')
            self.tcp_publisher = self.create_publisher(PoseStamped, get('tcp_pose_topic'), 10)
        self.previous = {}
        self.last_stamp = None
        self.subscription = self.create_subscription(
            JointState, get('joint_topic'), self.callback, qos_profile_sensor_data)
        self.get_logger().info(
            f'Actual pose: {self.base} -> {self.tip}; joints={self.fk.joint_names}. '
            f'TCP={self.tcp_frame or "disabled (link pose only)"}. '
            'JointState must contain all chain joints in radians, with nonzero timestamps.')

    def publish_pose(self, publisher, transform, stamp, key):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [transform[i][3] for i in range(3)]
        q = quaternion(transform)
        previous = self.previous.get(key)
        if previous is not None and sum(a*b for a, b in zip(previous, q)) < 0:
            q = [-v for v in q]
        self.previous[key] = q
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q
        publisher.publish(msg)

    def callback(self, msg):
        try:
            if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
                raise ValueError('JointState name/position lengths differ or names repeat')
            stamp = msg.header.stamp.sec*1_000_000_000 + msg.header.stamp.nanosec
            if stamp <= 0:
                raise ValueError('Missing measurement timestamp; fix JointState publisher')
            if self.last_stamp is not None and stamp <= self.last_stamp:
                raise ValueError('Duplicate/backward timestamp; restart node after replay clock reset')
            positions = dict(zip(msg.name, msg.position))
            missing = set(self.fk.joint_names) - positions.keys()
            if missing:
                raise ValueError(f'Incomplete JointState: missing {sorted(missing)}')
            if not all(math.isfinite(positions[name]) for name in self.fk.joint_names):
                raise ValueError('Nonfinite joint feedback')
            transform = self.fk.compute(positions)
            self.publish_pose(self.publisher, transform, msg.header.stamp, 'link')
            if self.tcp_publisher:
                self.publish_pose(self.tcp_publisher, multiply(transform, self.tcp), msg.header.stamp, 'tcp')
            self.last_stamp = stamp
        except (ValueError, KeyError) as exc:
            self.get_logger().warning(f'Skipping sample: {exc}', throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = None
    try:
        node = PoseCollector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
