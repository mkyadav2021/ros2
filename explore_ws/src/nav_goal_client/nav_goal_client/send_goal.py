#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped


class SendGoal(Node):
    def __init__(self):
        super().__init__('send_goal')
        # Action client for Nav2's navigate_to_pose action.
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def send(self, x, y, yaw_w=1.0, yaw_z=0.0):
        # Block until the Nav2 action server is up.
        self.get_logger().info('Waiting for navigate_to_pose action server...')
        self._client.wait_for_server()
        self.get_logger().info('Server available. Building goal.')

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'                       # goal is in the map frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        # Orientation is a quaternion. w=1, z=0 means "facing along map +X".
        goal.pose.pose.orientation.z = float(yaw_z)
        goal.pose.pose.orientation.w = float(yaw_w)

        self.get_logger().info(f'Sending goal: x={x}, y={y}')
        # send_goal_async returns a future for the goal handle; attach feedback cb.
        send_future = self._client.send_goal_async(
            goal, feedback_callback=self._on_feedback)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal REJECTED by server.')
            rclpy.shutdown()
            return
        self.get_logger().info('Goal ACCEPTED. Waiting for result...')
        # Now ask for the result; fires _on_result when navigation finishes.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        # distance_remaining is one of the feedback fields (meters).
        self.get_logger().info(
            f'Feedback: {fb.distance_remaining:.2f} m remaining')

    def _on_result(self, future):
        result = future.result().result
        status = future.result().status
        # In Jazzy the result has error_code (0 == NONE == success) and error_msg.
        self.get_logger().info(
            f'Navigation finished. status={status}, '
            f'error_code={result.error_code}, error_msg="{result.error_msg}"')
        rclpy.shutdown()


def main():
    rclpy.init()
    node = SendGoal()
    # Hard-coded goal.
    node.send(x=1.5, y=0.5)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
