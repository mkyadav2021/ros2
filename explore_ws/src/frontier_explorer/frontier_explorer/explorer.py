#!/usr/bin/env python3
import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        #tunable parameters
        self.min_cluster_size = 8       # cells; drop smaller frontier blobs as noise
        self.plan_period = 4.0          # seconds between exploration attempts
        self.blacklist_radius = 0.5     # m; goals within this of a blacklisted point are skipped
        self.max_failures = 2           # blacklist a frontier after this many hard failures

        #fix: ignore frontiers we're effectively standing on
        self.too_close_dist = 0.4       # m; a centroid this close can't be driven toward usefully

        #fix: catch the "reached but map didn't change" livelock
        self.repeat_tolerance = 0.4     # m; treat goals within this as "the same frontier"
        self.max_repeats = 3            # blacklist after selecting the same frontier this many times

        #state
        self.map_msg = None
        self.navigating = False
        self.blacklist = []             # list of (x, y) world points to avoid
        self.current_goal_xy = None
        self.goal_fail_counts = {}      # (rounded x, y) -> hard-failure count

        #fix 1 state
        self.last_goal_xy = None
        self.repeat_count = 0

        #map subscription (transient-local, matching slam_toolbox's latched map)
        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        #Nav2 action client (the Phase 3.3 mechanism)
        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        #TF for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #frontier marker publisher for RViz
        self.marker_pub = self.create_publisher(MarkerArray, '/frontiers', 10)

        #the exploration loop runs on a timer (not on every map msg)
        self.create_timer(self.plan_period, self._explore_step)
        self.get_logger().info('Frontier explorer started.')

    #callbacks
    def _on_map(self, msg):
        self.map_msg = msg

    #main loop
    def _explore_step(self):
        if self.map_msg is None:
            self.get_logger().info('Waiting for /map...')
            return
        if self.navigating:
            return  # let the current goal finish

        m = self.map_msg
        data, W, H = m.data, m.info.width, m.info.height
        ox, oy, res = (m.info.origin.position.x,
                       m.info.origin.position.y,
                       m.info.resolution)

        mask = self._frontier_mask(data, W, H)
        if not mask.any():
            self.get_logger().info('No frontier cells. Exploration complete.')
            self._publish_markers([])
            return

        clusters = self._cluster(mask)
        if not clusters:
            self.get_logger().info('No clusters above min size. Exploration complete.')
            self._publish_markers([])
            return

        # centroids in world coords
        candidates = [self._centroid_world(c, ox, oy, res) for c in clusters]
        self._publish_markers(candidates)  # show ALL frontiers in RViz

        # drop blacklisted candidates
        candidates = [p for p in candidates if not self._is_blacklisted(p)]
        if not candidates:
            self.get_logger().info('All frontiers blacklisted. Stopping.')
            return

        robot = self._robot_xy()
        if robot is None:
            self.get_logger().warn('No robot pose yet (TF map->base_link). Skipping.')
            return

        # fix: discard frontiers we're already sitting on
        reachable = [p for p in candidates
                     if math.hypot(p[0] - robot[0], p[1] - robot[1]) > self.too_close_dist]
        if not reachable:
            self.get_logger().info('Only on-top-of frontiers remain. Exploration complete.')
            return

        #nearest centroid to the robot
        target = min(reachable,
                     key=lambda p: math.hypot(p[0] - robot[0], p[1] - robot[1]))

        # fix: livelock guard (same frontier selected repeatedly without progress)
        if (self.last_goal_xy is not None and
                math.hypot(target[0] - self.last_goal_xy[0],
                           target[1] - self.last_goal_xy[1]) < self.repeat_tolerance):
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_goal_xy = target

        if self.repeat_count >= self.max_repeats:
            self.get_logger().warn(
                f'Frontier ({target[0]:.2f}, {target[1]:.2f}) selected '
                f'{self.repeat_count}x without progress; blacklisting.')
            self.blacklist.append(target)
            self.repeat_count = 0
            return

        self.get_logger().info(
            f'{len(clusters)} frontiers; driving to nearest at '
            f'({target[0]:.2f}, {target[1]:.2f})')
        self._send_goal(target)

    #frontier detection (vectorized)
    def _frontier_mask(self, data, W, H):
        """Boolean (H,W) mask: free cells (0) adjacent to unknown (-1), 4-connected."""
        grid = np.asarray(data, dtype=np.int16).reshape(H, W)
        free = (grid == 0)
        unknown = (grid == -1)
        unk_nbr = np.zeros_like(unknown)
        unk_nbr[1:, :]  |= unknown[:-1, :]   # unknown above
        unk_nbr[:-1, :] |= unknown[1:, :]    # unknown below
        unk_nbr[:, 1:]  |= unknown[:, :-1]   # unknown left
        unk_nbr[:, :-1] |= unknown[:, 1:]    # unknown right
        return free & unk_nbr

    def _cluster(self, mask):
        """Flood-fill (8-connected) True cells into clusters; drop clusters < min size.
        Returns list of clusters, each a list of (row, col)."""
        H, W = mask.shape
        visited = np.zeros_like(mask)
        clusters = []
        for sr, sc in np.argwhere(mask):
            if visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and mask[nr, nc] and not visited[nr, nc]):
                            visited[nr, nc] = True
                            q.append((nr, nc))
            if len(cells) >= self.min_cluster_size:
                clusters.append(cells)
        return clusters

    def _centroid_world(self, cells, ox, oy, res):
        rows = sum(r for r, _ in cells) / len(cells)
        cols = sum(c for _, c in cells) / len(cells)
        return (ox + (cols + 0.5) * res, oy + (rows + 0.5) * res)

    #robot pose via TF
    def _robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y)
        except TransformException:
            return None

    #blacklist
    def _is_blacklisted(self, p):
        return any(math.hypot(p[0] - bx, p[1] - by) < self.blacklist_radius
                   for bx, by in self.blacklist)

    def _record_failure(self, p):
        if p is None:
            return
        key = (round(p[0], 1), round(p[1], 1))
        self.goal_fail_counts[key] = self.goal_fail_counts.get(key, 0) + 1
        if self.goal_fail_counts[key] >= self.max_failures:
            self.blacklist.append(p)
            self.get_logger().warn(f'Blacklisted frontier near {key}.')

    #Nav2 goal
    def _send_goal(self, xy):
        if not self._nav.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Nav2 action server not available yet.')
            return
        self.navigating = True
        self.current_goal_xy = xy
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(xy[0])
        goal.pose.pose.position.y = float(xy[1])
        goal.pose.pose.orientation.w = 1.0
        fut = self._nav.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn('Goal rejected.')
            self._record_failure(self.current_goal_xy)
            self.navigating = False
            return
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        status = future.result().status
        if status == 4:  # STATUS_SUCCEEDED
            self.get_logger().info('Reached frontier.')
        else:
            self.get_logger().warn(f'Goal ended status={status}. Recording failure.')
            self._record_failure(self.current_goal_xy)
        self.navigating = False  #triggers a fresh frontier computation next timer tick

    #RViz markers
    def _publish_markers(self, points):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, (x, y) in enumerate(points):
            mk = Marker()
            mk.header.frame_id = 'map'
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = 'frontiers'
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = x
            mk.pose.position.y = y
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.25
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.0, 0.0, 1.0
            arr.markers.append(mk)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = FrontierExplorer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
