import datetime
import pickle
from collections import deque
from typing import Tuple, Dict

import numpy as np
import rospy
from matplotlib import pyplot as plt
from tf import transformations

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped

from agents.mapping import Env
from environment.nav_utilities.coordinates_converter import cvt_polar_to_cartesian


# from environment.nav_utilities.state_helper import visualize_cartesian


class RosRobot:
    def __init__(self, args):
        self.visible_zone_limit = 3.5
        self.args = args
        self._x = None
        self._y = None
        # [-np.pi, np.pi]
        self._yaw = None
        self._current_v = None
        self._current_w = None
        self._ranges = None
        self.range_max = None
        self.ray_min_angle = None
        self.ray_max_angle = None
        self.angle_resolution = None

        self.history_trajectory = []
        self.history_applied_v = []
        self.history_applied_w = []
        self.history_real_v = []
        self.history_real_w = []

        self.count = 0
        self.env_path = None
        # subscribe odom and scan pipe
        rospy.Subscriber("/scan", LaserScan, callback=self.scan_callback)
        rospy.Subscriber("/odom", Odometry, callback=self.odom_callback)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, callback=self.pose_callback)

        # publisher
        if self.args.env == Env.Real:
            topic = "/cmd_vel_mux/input/safety_controller"
        elif self.args.env == Env.Gazebo:
            topic = "/cmd_vel"
        else:
            raise NotImplementedError
        self.action_pub = rospy.Publisher(topic, Twist, queue_size=3)
        self.seq_len = 4
        self.hit_vector_list = deque(maxlen=self.seq_len)
        self.polar_positions_list = deque(maxlen=self.seq_len)
        self.cartesian_coordinates_list = deque(maxlen=self.seq_len)
        # todo 确定三种环境环境的次数

    def scan_callback(self, msg: LaserScan):
        # scan once every 0.1s
        self.ray_min_angle = msg.angle_min
        self.ray_max_angle = msg.angle_max
        self.angle_resolution = msg.angle_increment
        self.range_max = msg.range_max

        ranges = np.array(msg.ranges, dtype=np.float32)
        # ranges = np.flip(ranges)
        # 1440
        ranges = ranges[::4]
        # hit_vector
        # polar_positions
        # cartesian_positions
        # [0, 2 * pi]
        hit_thetas = (np.linspace(0, 2 * np.pi, len(ranges), endpoint=False) + np.pi / 2) % (2 * np.pi)
        # [0, visible_zone_limit]
        # clip hit distances with upper self.visible_zone_limit
        hit_distances = np.where(ranges <= self.visible_zone_limit, ranges, self.visible_zone_limit)
        # normalize
        hit_fractions = hit_distances / self.visible_zone_limit
        hit_vector = np.where(ranges <= self.visible_zone_limit, 1, 0)
        polar_positions = np.array([hit_thetas, hit_fractions]).transpose()
        # relative cartesian coordinates
        cartesian_coordinates = cvt_polar_to_cartesian(polar_positions)

        self.polar_positions_list.append(polar_positions)
        self.cartesian_coordinates_list.append(cartesian_coordinates)
        self.hit_vector_list.append(hit_vector)
        # visualize_cartesian(cartesian_coordinates[:, 0], cartesian_coordinates[:, 1],
        #                     "cvt_polar_positions_to_reference")

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        # pose
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._yaw = transformations.euler_from_quaternion(
            (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            )
        )[2]

        now = datetime.datetime.now()
        otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
        print("时间：{}，AMCL 坐标：({},{},{})".format(otherStyleTime, self._x, self._y, self._yaw))

    def odom_callback(self, msg: Odometry):

        # current velocity
        self._current_v = np.sqrt(np.square(msg.twist.twist.linear.x) + np.square(msg.twist.twist.linear.x))
        self._current_w = msg.twist.twist.angular.z

        self.history_trajectory.append([self._x, self._y])
        self.count += 1

    def get_x_y_yaw(self) -> np.ndarray:
        return np.array([self._x, self._y, self._yaw])

    def get_position(self) -> np.ndarray:
        return np.array([self._x, self._y])

    def get_x_y(self) -> np.ndarray:
        return np.array([self._x, self._y])

    def get_yaw(self) -> float:
        return self._yaw

    def get_v_w(self) -> Tuple[float, float]:
        return self._current_v, self._current_w

    def stop(self):
        self.apply_action(0, 0)

    def apply_action(self, v: float, w: float):
        # print("v={}, w={}".format(v, w))
        # TODO 调整倍数，保证速度在0-0.5
        v_limit = 0.7
        w_limit = 1.2

        msg = Twist()
        msg.linear.x = v * v_limit
        msg.angular.z = w * w_limit
        self.action_pub.publish(msg)

        self.history_applied_v.append(v)
        self.history_applied_w.append(w)
        self.history_real_v.append(self._current_v)
        self.history_real_w.append(self._current_w)

    def lidar_scan(self):
        dists = [1 if dist == np.Inf else dist / self.range_max for dist in self._ranges]
        return np.array(dists)

    def is_collide(self):
        return np.min(self._ranges) < 0.14


def visualize_cartesian(xs, ys, title="Plot lidar cartesian positions", c='r'):
    plt.scatter(xs, ys, lw=2, c=c)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    rospy.init_node("ros_robot")
    robot = RosRobot()
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        robot.apply_action(-1, 0)
        rate.sleep()
