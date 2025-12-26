#!/usr/bin/env python3
"""
Robot Piano Player Library - robot_pianist_library.py
Provides a class with high-level primitives to control the robot pianist.
This library is intended to be imported by execution scripts.
"""
import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Movel, Handangle
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
import numpy as np
import time

class PianoKeyboard:
    """UPDATED: Internal helper class to store piano layout and dynamic finger mappings."""
    def __init__(self):
        self.key_spacing = 0.023  # meters
        
        # UPDATED: This now defines the physical shift for each named position.
        self.hand_positions = {
            'C4-F4': 0.0,
            'E4-A4': -2 * self.key_spacing  # Shift right by 2 keys
        }

        # NEW: Dynamic finger mapping based on hand position.
        # This is the core of the fix. Each position has its own map.
        self.hand_position_maps = {
            'C4-F4': {
                'C4': 'index', 'D4': 'middle', 'E4': 'ring', 'F4': 'pinky'
            },
            'E4-A4': {
                'E4': 'index', 'F4': 'middle', 'G4': 'ring', 'A4': 'pinky'
            }
        }

        self.servo_values = {
            'open': [0, 0, 0, 0, 0, 0],
            'index_press': [0, 6800, 0, 0, 0, 0],
            'middle_press': [0, 0, 6300, 0, 0, 0],
            'ring_press': [0, 0, 0, 8000, 0, 0],
            'pinky_press': [0, 0, 0, 0, 12000, 0],
            'index_lifted': [0, 100, 0, 0, 0, 0],
            'middle_lifted': [0, 0, 100, 0, 0, 0],
            'ring_lifted': [0, 0, 0, 100, 0, 0],
            'pinky_lifted': [0, 0, 0, 0, 100, 0],
        }

class RobotPianoPlayer(Node):
    """Provides the motion primitive API to control the robot pianist."""

    def __init__(self):
        super().__init__('robot_piano_player_library_node')
        self.pose_pub = self.create_publisher(Movel, '/right_arm_controller/rm_driver/movel_cmd', 10)
        self.hand_pub = self.create_publisher(Handangle, '/right_arm_controller/rm_driver/set_hand_follow_pos_cmd', 10)
        self.stop_pub = self.create_publisher(Bool, '/right_arm_controller/rm_driver/move_stop_cmd', 10)
        self.create_subscription(Pose, '/right_arm_controller/rm_driver/udp_arm_position', self._robot_pose_callback, 10)
        
        self.piano = PianoKeyboard()
        self.robot_base_position = None
        self.robot_base_orientation = None
        self.robot_current_pose = None
        self.robot_calibrated = False
        self.current_wrist_offset = 0.0
        self.NOTE_SEPARATION_DELAY = 0.05
        
        # NEW: Track the current hand position NAME, not just the offset
        self.current_hand_position = None
        self.last_played_note = None
        self.CONSECUTIVE_KEY_LIFT_TIME = 0.15

        self.get_logger().info("Robot Piano Player Library Initialized.")

    def _robot_pose_callback(self, msg: Pose):
        self.robot_current_pose = {
            'position': np.array([msg.position.x, msg.position.y, msg.position.z]),
            'orientation': np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        }

    def calibrate_piano_base_position(self, wait_for_input=True):
        """A blocking call that initializes the robot's position over the piano."""
        self.get_logger().info("Waiting for robot pose...")
        while self.robot_current_pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if wait_for_input:
            self.get_logger().info("Position robot fingers over the 'C4-F4' home position.")
            input("Press Enter when ready...")
            
        self.robot_base_position = self.robot_current_pose['position'].copy()
        self.robot_base_orientation = self.robot_current_pose['orientation'].copy()
        self.robot_calibrated = True
        
        # NEW: Set the initial hand position state
        self.current_hand_position = 'C4-F4'
        self.current_wrist_offset = self.piano.hand_positions['C4-F4']
        self.last_played_note = None

        self._send_finger_command(self.piano.servo_values['open'])
        self.get_logger().info(f"Piano calibrated at home position: {self.robot_base_position}")
        return True

    def move_wrist_to_position(self, position_name: str):
        """A blocking call that moves the robot's arm to a named location."""
        if position_name not in self.piano.hand_positions:
            self.get_logger().error(f"Unknown hand position: '{position_name}'")
            return False
            
        # UPDATED: Check against the stored position name
        if self.current_hand_position == position_name:
            self.get_logger().info(f"Wrist already at '{position_name}'. No move needed.")
            return True
            
        self.get_logger().info(f"Moving wrist to position: '{position_name}'...")
        self._send_finger_command(self.piano.servo_values['open'])
        
        target_offset = self.piano.hand_positions[position_name]
        target_position = self.robot_base_position.copy()
        target_position[0] += target_offset
        
        if self._send_wrist_position_and_wait(target_position):
            # NEW: Update BOTH the offset and the position name
            self.current_wrist_offset = target_offset
            self.current_hand_position = position_name
            self.last_played_note = None # Reset last note after moving
            time.sleep(0.3)
            return True
        return False

    def play_key(self, note: str, duration: float):
        """UPDATED: A blocking call that uses the current hand position for correct finger mapping."""
        if not self.current_hand_position:
            self.get_logger().error("Cannot play key, current hand position is unknown.")
            return False
            
        # NEW: Get the correct finger map for the CURRENT hand position
        current_finger_map = self.piano.hand_position_maps[self.current_hand_position]
        
        # NEW: Check if the note is playable from this position
        if note not in current_finger_map:
            self.get_logger().error(f"Cannot play note '{note}' from current hand position '{self.current_hand_position}'.")
            return False
            
        finger_to_use = current_finger_map[note]
        servo_command = f"{finger_to_use}_press"
        servo_values = self.piano.servo_values[servo_command]
        
        is_consecutive_same_key = (self.last_played_note == note)
        
        if is_consecutive_same_key:
            self.get_logger().info(f"Consecutive key detected: {note}. Lifting for separation.")
            lifted_command = f"{finger_to_use}_lifted"
            lifted_values = self.piano.servo_values[lifted_command]
            self._send_finger_command(lifted_values)
            time.sleep(self.CONSECUTIVE_KEY_LIFT_TIME)
        
        self.get_logger().info(f"Playing {note} with {finger_to_use} finger for {duration:.2f}s")
        self._send_finger_command(servo_values)
        time.sleep(duration)
        self._send_finger_command(self.piano.servo_values['open'])
        time.sleep(self.NOTE_SEPARATION_DELAY)
        
        self.last_played_note = note
        return True
    
    def return_to_base(self):
        """A blocking call that moves the robot to a safe, neutral position."""
        self.get_logger().info("Returning to base position...")
        self._send_finger_command(self.piano.servo_values['open'])
        time.sleep(0.2)
        if self.robot_base_position is not None:
             self._send_wrist_position_and_wait(self.robot_base_position)
             self.current_wrist_offset = 0.0
             # NEW: Reset state on return to base
             self.current_hand_position = 'C4-F4'
             self.last_played_note = None
        self.get_logger().info("Robot is at base.")

    def emergency_stop(self):
        """A dedicated safe-shutdown call for handling Ctrl+C."""
        self.get_logger().info("--- EMERGENCY STOP ---")
        self.get_logger().info("Publishing immediate move_stop command.")
        self.stop_pub.publish(Bool(data=True))
        time.sleep(0.1)
        self.get_logger().info("Opening hand and returning to base position immediately.")
        self.return_to_base()

    def _send_wrist_position_and_wait(self, position, timeout=5.0):
        # Internal helper to send move command and wait for completion
        msg = Movel()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = float(position[0]), float(position[1]), float(position[2])
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = float(self.robot_base_orientation[0]), float(self.robot_base_orientation[1]), float(self.robot_base_orientation[2]), float(self.robot_base_orientation[3])
        msg.speed, msg.block = 100, False
        self.pose_pub.publish(msg)
        start_time, tolerance = time.time(), 0.005
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.robot_current_pose is not None and np.linalg.norm(self.robot_current_pose['position'] - position) < tolerance:
                return True
        self.get_logger().warn("Wrist move timed out, but continuing.")
        return True

    def _send_finger_command(self, servo_values):
        """UPDATED: Internal helper to send finger command with proper integer conversion and error handling."""
        try:
            msg = Handangle()
            # This logic correctly handles the conversion for the robot's hardware.
            msg.hand_angle = [int(x) if x < 32768 else int(x) - 65536 for x in servo_values]
            msg.block = False
            self.hand_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error sending finger command: {e}")
