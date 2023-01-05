import copy

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class RosDebugger:
    def __init__(self):
        self.marker_pub = rospy.Publisher("trajectory", Marker, queue_size=10)

    def draw_lines(self, env_path: np.ndarray):
        lines, map_points = MarkerArray(), MarkerArray()
        length = len(env_path)
        for i in range(0, length - 1, 2):
            line_strip, points = Marker(), Marker()

            line_strip.header.frame_id = points.header.frame_id = "odom"
            line_strip.header.stamp = points.header.stamp = rospy.Time().now()
            line_strip.ns = points.ns = "points_and_lines"
            line_strip.action = points.action = Marker.ADD
            # line_strip.pose.orientation.w = 1.0
            line_strip.id = i
            line_strip.type = Marker.LINE_STRIP
            line_strip.scale.x = 0.1
            line_strip.color.g = 1.0
            line_strip.color.a = 1.0

            points.id = 0
            points.type = Marker.POINTS
            points.scale.x = 2
            points.scale.y = 2
            points.color.r = 1.0
            points.color.a = 1.0

            # line_strip obtains points
            p, q = Point(), Point()
            p.x = env_path[i, 0]
            p.y = env_path[i, 1]
            p.z = 0.5
            q.x = env_path[i + 1, 0]
            q.y = env_path[i + 1, 1]
            q.z = 0.5

            line_strip.points.append(p)
            line_strip.points.append(q)
            points.points.append(p)
            points.points.append(q)

            lines.markers.append(line_strip)
            map_points.markers.append(points)

        self.marker_pub.publish(lines)
        self.marker_pub.publish(map_points)

    def draw_lines2(self, env_path):
        line_list = Marker()

        line_list.header.frame_id = "odom"
        line_list.header.stamp = rospy.Time.now()
        line_list.ns = "lines"
        line_list.action = Marker.ADD
        line_list.pose.orientation.w = 1.0
        line_list.id = 2
        line_list.type = Marker.LINE_LIST
        line_list.scale.x = 0.1
        line_list.color.r = 1.0
        line_list.color.a = 1.0

        for waypoint in env_path:
            p = Point()
            p.x = waypoint[0]
            p.y = waypoint[1]
            p.z = 0

            line_list.points.append(p)
            p = copy.deepcopy(p)
            p.z += 1.0
            line_list.points.append(p)
        self.marker_pub.publish(line_list)
