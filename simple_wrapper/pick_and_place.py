#!/usr/bin/env python3
from threading import Thread
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from pymoveit2 import MoveIt2, MoveIt2State, GripperInterface
from pymoveit2.robots import fr3 as robot


class MoveItFranka:
    def __init__(self, node: Node):
        self.node = node
        self.callback_group = ReentrantCallbackGroup()

        # MoveIt2 Interface for Arm
        self.moveit2 = MoveIt2(
            node=node,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )

        # Gripper Interface
        self.gripper = GripperInterface(
            node=node,
            gripper_joint_names=robot.gripper_joint_names(),
            open_gripper_joint_positions=robot.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=robot.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=robot.MOVE_GROUP_GRIPPER,
            callback_group=self.callback_group,
            gripper_command_action_name="gripper_action_controller/gripper_cmd",
        )

        # Default parameters
        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.moveit2.max_velocity = 0.5
        self.moveit2.max_acceleration = 0.5
        self.moveit2.cartesian_avoid_collisions = False
        self.moveit2.cartesian_jump_threshold = 0.0

    # Arm movement functions
    # Move to a specific pose
    def move_to_pose(self, position, quat_xyzw, cartesian=False, cartesian_max_step=0.0025, synchronous=True, cancel_after_secs=0.0):
        self.node.get_logger().info(f"Moving to pose {position} with orientation {quat_xyzw}")
        self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw, cartesian=cartesian, cartesian_max_step=cartesian_max_step)
        if synchronous:
            self.moveit2.wait_until_executed()
        else:
            self._asynchronous_move(cancel_after_secs)

    # Move to a specific joint configuration
    def move_to_configuration(self, joint_positions, synchronous=True, cancel_after_secs=0.0):
        self.node.get_logger().info(f"Moving to joint configuration: {joint_positions}")
        self.moveit2.move_to_configuration(joint_positions)
        if synchronous:
            self.moveit2.wait_until_executed()
        else:
            self._asynchronous_move(cancel_after_secs)

    # Gripper control functions
    def open_gripper(self):
        self.node.get_logger().info("Opening gripper")
        self.gripper.open()
        self.gripper.wait_until_executed()

    def close_gripper(self):
        self.node.get_logger().info("Closing gripper")
        self.gripper.close()
        self.gripper.wait_until_executed()

    def toggle_gripper(self):
        self.node.get_logger().info("Toggling gripper")
        self.gripper()
        self.gripper.wait_until_executed()

    def move_gripper_to_position(self, position: float):
        """
        Move the gripper to a specific position.
        - `position` should be within the valid range of the gripper.
        """
        self.node.get_logger().info(f"Moving gripper to position {position}")
        self.gripper.move_to_position(position)
        self.gripper.wait_until_executed()

    # Internal function for async execution
    def _asynchronous_move(self, cancel_after_secs):
        rate = self.node.create_rate(10)
        while self.moveit2.query_state() != MoveIt2State.EXECUTING:
            rate.sleep()

        future = self.moveit2.get_execution_future()

        if cancel_after_secs > 0.0:
            sleep_time = self.node.create_rate(cancel_after_secs)
            sleep_time.sleep()
            self.moveit2.cancel_execution()

        while not future.done():
            rate.sleep()

        self.node.get_logger().info(f"Result status: {future.result().status}")
        self.node.get_logger().info(f"Result error code: {future.result().result.error_code}")


# Example usage
def main():
    rclpy.init()
    node = Node("moveit_franka_node")

    moveit_franka = MoveItFranka(node)

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    node.create_rate(1.0).sleep()

    # Simple Pick and Place
    ############################################################
    # Define poses
    initial_pose = {
        "position": [0.5, 0.0, 0.25],
        "quat_xyzw": [1.0, 0.0, 0.0, 0.0],
        "cartesian": False
    }
    approaching_pose = {
        "position": [0.5, 0.0, 0.2],
        "quat_xyzw": [1.0, 0.0, 0.0, 0.0],
        "cartesian": True
    }
    grasping_pose = {
        "position": [0.5, 0.0, 0.15],
        "quat_xyzw": [1.0, 0.0, 0.0, 0.0],
        "cartisian": True
    }
    retrieving_pose = {
        "position": [0.5, 0.0, 0.25],
        "quat_xyzw": [1.0, 0.0, 0.0, 0.0],
        "cartisian": True
    }
    placing_pose = {
        "position": [0.4, 0.0, 0.25],
        "quat_xyzw": [1.0, 0.0, 0.0, 0.0],
        "cartisian": True
    }
    ############################################################


    # Move to initial pose
    moveit_franka.move_to_pose(**initial_pose)

    # Move to approaching pose
    moveit_franka.move_to_pose(**approaching_pose)

    # Move to grasping pose
    moveit_franka.move_to_pose(**grasping_pose)

    # Close gripper to grasp
    moveit_franka.close_gripper()

    # Move to retrieving pose
    moveit_franka.move_to_pose(**retrieving_pose)

    # Move to placing pose
    moveit_franka.move_to_pose(**placing_pose)

    # Open gripper to place
    moveit_franka.open_gripper()

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
