#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass

import rospy
from sensor_msgs.msg import Joy, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from tf.transformations import euler_from_quaternion

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spline_lookup import SplineLookupTable

def clamp(x, x_min, x_max):
    return max(x_min, min(x_max, x))


@dataclass
class PID:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    i_min: float = -1.0
    i_max: float = 1.0

    integ: float = 0.0
    prev_err: float = None

    def reset(self):
        self.integ = 0.0
        self.prev_err = None

    def update(self, error, dt):
        if dt <= 0.0:
            return self.kp * error  # ignore I/D if dt bad

        # Integral term
        self.integ += error * dt
        self.integ = clamp(self.integ, self.i_min, self.i_max)

        # Derivative term
        if self.prev_err is None:
            d = 0.0
        else:
            d = (error - self.prev_err) / dt
        self.prev_err = error

        return self.kp * error + self.ki * self.integ + self.kd * d


class VertiWheelControlNode:
    MODE_SAFE = 0
    MODE_MANUAL = 1
    MODE_AUTO = 2

    CAM_MODE_FIXED = 0
    CAM_MODE_STABILIZED = 1

    def __init__(self):
        rospy.init_node("vertiwheeler_control_node")

        self.neg_angle_lookup = SplineLookupTable.load(os.path.dirname(os.path.abspath(__file__))+"/v4w1_neg_steering_calibration")
        self.pos_angle_lookup = SplineLookupTable.load(os.path.dirname(os.path.abspath(__file__))+"/v4w1_pos_steering_calibration")
        self.throttle_lookup = SplineLookupTable.load(os.path.dirname(os.path.abspath(__file__))+"/v4w1_linear_calibration")

        # ===== Parameters =====
        # Control loop
        self.control_rate = rospy.get_param("~control_rate", 50.0)
        self.auto_cmd_timeout = rospy.get_param("~auto_cmd_timeout", 0.5)  # s
        self.deadzone = rospy.get_param("~deadzone", 0.05)

        # Velocity / acceleration limits (same units as cmd_vel.linear.x)
        self.max_vel = rospy.get_param("~v4w_control/max_vel", 3.0)
        self.min_vel = rospy.get_param("~v4w_control/min_vel", -3.0)
        self.max_acc = rospy.get_param("~v4w_control/max_acc", 3.0)   # m/s^2
        self.min_acc = rospy.get_param("~v4w_control/min_acc", -3.0)  # m/s^2

        # Steering limits (steering command units, e.g. normalized [-1,1] or radians)
        self.max_str = rospy.get_param("~v4w_control/max_str", 1.0)
        self.min_str = rospy.get_param("~v4w_control/min_str", -1.0)
        self.steer_trim_step = rospy.get_param("~v4w_control/steer_trim_step", 0.05)
        self.steer_trim_limit = rospy.get_param("~v4w_control/steer_trim_limit", 0.3)

        # Camera parameters (angle units arbitrary but consistent)
        cam_kp = rospy.get_param("~camera_stabilizer/cam_kp", 1.0)
        cam_ki = rospy.get_param("~camera_stabilizer/cam_ki", 0.0)
        cam_kd = rospy.get_param("~camera_stabilizer/cam_kd", 0.0)
        cam_i_max = rospy.get_param("~camera_stabilizer/cam_i_max", 0.5)
        self.cam_reset_angle = rospy.get_param("~camera_stabilizer/cam_reset_angle", 0.0)
        self.cam_min_angle = rospy.get_param("~camera_stabilizer/cam_min_angle", -0.7)
        self.cam_max_angle = rospy.get_param("~camera_stabilizer/cam_max_angle", 0.7)
        self.cam_step = rospy.get_param("~camera_stabilizer/cam_step", 0.05)

        self.cam_pid = PID(
            kp=cam_kp,
            ki=cam_ki,
            kd=cam_kd,
            i_min=-cam_i_max,
            i_max=cam_i_max,
        )

        # Initial diff/gear state
        self.gear = rospy.get_param("~v4w_control/gear_state", 0)  # 0 = low, 1 = high
        self.front_diff = rospy.get_param("~v4w_control/front_diff_state", 0)  # 0 = unlocked, 1 = locked
        self.rear_diff = rospy.get_param("~v4w_control/rear_diff_state", 0)  # 0 = unlocked, 1 = locked

        # Joystick mapping (PS4 defaults, override with params if needed)
        self.axis_steer = rospy.get_param("~v4w_control/axis_steer", 0)  # left stick LR
        self.axis_throttle = rospy.get_param("~v4w_control/axis_throttle", 4)  # right stick UD
        self.axis_dpad_x = rospy.get_param("~v4w_control/axis_dpad_x", 6)
        self.axis_dpad_y = rospy.get_param("~v4w_control/axis_dpad_y", 7)

        self.fd_lock = rospy.get_param("~v4w_control/btn_square", 0)
        self.fd_unlock = rospy.get_param("~v4w_control/btn_cross", 1)
        self.rd_unlock = rospy.get_param("~v4w_control/btn_circle", 2)
        self.rd_lock = rospy.get_param("~v4w_control/btn_triangle", 3)
        self.gear_low = rospy.get_param("~v4w_control/btn_l1", 4)
        self.gear_high = rospy.get_param("~v4w_control/btn_l2", 5)
        self.disarm = rospy.get_param("~v4w_control/btn_share", 8)
        self.arm = rospy.get_param("~v4w_control/btn_options", 9)
        self.mode_switch = rospy.get_param("~v4w_control/btn_l3", 10)
        self.cam_stb_sw = rospy.get_param("~v4w_control/btn_r3", 11)
        self.cam_reset = rospy.get_param("~v4w_control/btn_ps", 12)

        # Safe command on startup / lock
        self.safe_steering = -0.75
        self.safe_throttle = 0.0
        self.safe_gear = 0.0
        self.safe_front_diff = 0.0
        self.safe_rear_diff = 0.0
        self.safe_cam = 0.0

        # ===== State =====
        self.mode = self.MODE_SAFE
        self.controls_unlocked = False

        self.steering = self.safe_steering
        self.throttle = self.safe_throttle
        self.steer_trim = 0.0

        self.camera_mode = self.CAM_MODE_FIXED
        self.camera_angle = self.cam_reset_angle
        self.cam_setpoint = self.cam_reset_angle

        self.pitch = None  # from IMU
        self.last_cmd_vel = None
        self.last_cmd_vel_time = rospy.Time(0)

        self.last_joy_buttons = []
        self.last_joy_axes = []

        self.last_loop_time = rospy.Time.now()

        # ===== ROS I/O =====
        self.pub_control = rospy.Publisher("control", Float32MultiArray, queue_size=1)
        self.pub_real_cmd = rospy.Publisher("~real_cmd_vel", Twist, queue_size=1)

        rospy.Subscriber("joy", Joy, self.joy_callback, queue_size=1)
        rospy.Subscriber("cmd_vel", Twist, self.cmd_vel_callback, queue_size=1)
        rospy.Subscriber("imu", Imu, self.imu_callback, queue_size=1)

        rospy.loginfo("VertiWheel control node started in SAFE mode.")
        self.spin()

    # --------- Callbacks ---------

    def joy_callback(self, msg):
        # Initialize last button/axis states if first time
        if not self.last_joy_buttons:
            self.last_joy_buttons = list(msg.buttons)
        if not self.last_joy_axes:
            self.last_joy_axes = list(msg.axes)

        self.handle_joy_buttons(msg)
        self.handle_joy_axes(msg)

        self.last_joy_buttons = list(msg.buttons)
        self.last_joy_axes = list(msg.axes)

    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = msg
        self.last_cmd_vel_time = rospy.Time.now()

    def imu_callback(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, pitch, _ = euler_from_quaternion(quat)
        self.pitch = pitch

    # --------- Helpers for input handling ---------

    def button_pressed(self, buttons, last_buttons, idx):
        if idx < 0 or idx >= len(buttons) or idx >= len(last_buttons):
            return False
        return buttons[idx] == 1 and last_buttons[idx] == 0

    def axis_positive_edge(self, axes, last_axes, idx, thresh=0.5):
        if idx < 0 or idx >= len(axes) or idx >= len(last_axes):
            return False
        return axes[idx] > thresh and last_axes[idx] <= thresh

    def axis_negative_edge(self, axes, last_axes, idx, thresh=-0.5):
        if idx < 0 or idx >= len(axes) or idx >= len(last_axes):
            return False
        return axes[idx] < thresh and last_axes[idx] >= thresh

    # --------- JOY handling (discrete state) ---------

    def handle_joy_buttons(self, msg):
        b = msg.buttons
        lb = self.last_joy_buttons

        # Options: unlock and go to MANUAL
        if self.button_pressed(b, lb, self.arm):
            self.controls_unlocked = True
            self.mode = self.MODE_MANUAL
            rospy.loginfo("Controls UNLOCKED: MANUAL (joy) mode.")

        # Share: lock and go back to SAFE
        if self.button_pressed(b, lb, self.disarm):
            self.controls_unlocked = False
            self.mode = self.MODE_SAFE
            rospy.loginfo("Controls LOCKED: SAFE mode.")

        if not self.controls_unlocked:
            # ignore the rest when locked
            return

        # L3 press: toggle AUTO <-> MANUAL
        if self.button_pressed(b, lb, self.mode_switch):
            if self.mode == self.MODE_AUTO:
                self.mode = self.MODE_MANUAL
                rospy.loginfo("Switched to MANUAL (joy) mode.")
            else:
                self.mode = self.MODE_AUTO
                rospy.loginfo("Switched to AUTO (cmd_vel) mode.")

        # Gear: L1 = low, L2 = high
        if self.button_pressed(b, lb, self.gear_low):
            self.gear = 0.0
            rospy.loginfo("Gear: LOW")

        if self.button_pressed(b, lb, self.gear_high):
            self.gear = 1.0
            rospy.loginfo("Gear: HIGH")

        # Front diff: square = lock, cross = unlock
        if self.button_pressed(b, lb, self.fd_lock):
            self.front_diff = 1.0
            rospy.loginfo("Front diff: LOCKED")

        if self.button_pressed(b, lb, self.fd_unlock):
            self.front_diff = 0.0
            rospy.loginfo("Front diff: UNLOCKED")

        # Rear diff: triangle = lock, circle = unlock
        if self.button_pressed(b, lb, self.rd_lock):
            self.rear_diff = 1.0
            rospy.loginfo("Rear diff: LOCKED")

        if self.button_pressed(b, lb, self.rd_unlock):
            self.rear_diff = 0.0
            rospy.loginfo("Rear diff: UNLOCKED")

        # PS button: reset camera angle/setpoint and PID
        if self.button_pressed(b, lb, self.cam_reset):
            self.camera_angle = self.cam_reset_angle
            self.cam_setpoint = self.cam_reset_angle
            self.cam_pid.reset()
            rospy.loginfo("Camera reset to default angle.")

        # R3: toggle camera mode fixed <-> stabilized
        if self.button_pressed(b, lb, self.cam_stb_sw):
            if self.camera_mode == self.CAM_MODE_FIXED:
                self.camera_mode = self.CAM_MODE_STABILIZED
                # Use current pitch as setpoint if available, else reset
                if self.pitch is not None:
                    self.cam_setpoint = self.pitch
                else:
                    self.cam_setpoint = self.cam_reset_angle
                self.cam_pid.reset()
                rospy.loginfo("Camera mode: STABILIZED")
            else:
                self.camera_mode = self.CAM_MODE_FIXED
                # Keep current command as fixed angle
                self.cam_pid.reset()
                rospy.loginfo("Camera mode: FIXED")

    def handle_joy_axes(self, msg):
        a = msg.axes
        la = self.last_joy_axes

        # D-pad up/down: camera angle or setpoint increment/decrement
        if self.axis_positive_edge(a, la, self.axis_dpad_y):
            # up
            if self.camera_mode == self.CAM_MODE_STABILIZED:
                self.cam_setpoint = clamp(
                    self.cam_setpoint + self.cam_step, self.cam_min_angle, self.cam_max_angle
                )
            else:
                self.camera_angle = clamp(
                    self.camera_angle + self.cam_step, self.cam_min_angle, self.cam_max_angle
                )
        if self.axis_negative_edge(a, la, self.axis_dpad_y):
            # down
            if self.camera_mode == self.CAM_MODE_STABILIZED:
                self.cam_setpoint = clamp(
                    self.cam_setpoint - self.cam_step, self.cam_min_angle, self.cam_max_angle
                )
            else:
                self.camera_angle = clamp(
                    self.camera_angle - self.cam_step, self.cam_min_angle, self.cam_max_angle
                )

        # D-pad left/right: steering trim
        if self.axis_positive_edge(a, la, self.axis_dpad_x):
            # right
            self.steer_trim += self.steer_trim_step
        if self.axis_negative_edge(a, la, self.axis_dpad_x):
            # left
            self.steer_trim -= self.steer_trim_step

        self.steer_trim = clamp(
            self.steer_trim, -self.steer_trim_limit, self.steer_trim_limit
        )

    # --------- Control computations ---------

    def joy_to_commands(self, axes):
        """Map joystick axes to (steering, throttle_raw) before accel limiting."""
        steering = 0.0
        throttle = 0.0

        if self.axis_steer < len(axes):
            steer_val = axes[self.axis_steer]
            if abs(steer_val) < self.deadzone:
                steer_val = 0.0
            steering = steer_val * self.max_str

        if self.axis_throttle < len(axes):
            thr_val = axes[self.axis_throttle]
            # On PS4, up = -1, down = +1; invert so pushing up is forward (+)
            thr_val = -thr_val
            if abs(thr_val) < self.deadzone:
                thr_val = 0.0

            # Map [-1, 1] to [min_vel, max_vel], allowing asymmetric
            if thr_val >= 0.0:
                throttle = thr_val * self.max_vel
            else:
                throttle = thr_val * abs(self.min_vel)

        steering = clamp(steering + self.steer_trim, self.min_str, self.max_str)
        return steering, throttle

    def cmd_vel_to_commands(self, cmd_vel):
        """Map cmd_vel to (steering, throttle_raw) before accel limiting."""
        if cmd_vel is None:
            return 0.0, 0.0

        v = clamp(cmd_vel.linear.x, self.min_vel, self.max_vel)
        throttle_robot = self.throttle_lookup.speed_to_cmd(v)
        # Assuming angular.z is already in steering units, just clamp
        steering = clamp(cmd_vel.angular.z, self.min_str, self.max_str)
        steering = clamp(steering + self.steer_trim, self.min_str, self.max_str)
        if steering < 0.0:
            steering_robot = self.neg_angle_lookup.speed_to_cmd(steering)
        else:
            steering_robot = self.pos_angle_lookup.speed_to_cmd(steering)
        return steering_robot, throttle_robot

    def apply_accel_limit(self, desired_throttle, dt):
        """Limit acceleration/deceleration for throttle channel."""
        if dt <= 0.0:
            return desired_throttle

        dv = desired_throttle - self.throttle
        if dv > 0.0:
            max_dv = self.max_acc * dt
            dv = min(dv, max_dv)
        else:
            max_dv = self.min_acc * dt  # negative
            dv = max(dv, max_dv)
        return self.throttle + dv

    def update_camera(self, dt):
        """Update camera_angle depending on mode and IMU."""
        if self.camera_mode == self.CAM_MODE_FIXED:
            # Nothing to do; camera_angle is directly controlled via D-pad / reset
            return

        if self.pitch is None:
            return

        # Stabilized: drive pitch -> setpoint using PID; PID output directly used as camera angle
        error = self.cam_setpoint - self.pitch
        out = self.cam_pid.update(error, dt)
        self.camera_angle = clamp(out, self.cam_min_angle, self.cam_max_angle)

    # --------- Main loop ---------

    def spin(self):
        rate = rospy.Rate(self.control_rate)
        self.last_loop_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last_loop_time).to_sec()
            self.last_loop_time = now

            # Auto mode timeout: fallback to MANUAL
            if (
                self.mode == self.MODE_AUTO
                and (now - self.last_cmd_vel_time).to_sec() > self.auto_cmd_timeout
            ):
                self.mode = self.MODE_MANUAL
                rospy.logwarn_throttle(
                    5.0,
                    "AUTO mode timeout (no cmd_vel). Falling back to MANUAL (joy).",
                )

            # Compute steering/throttle
            if not self.controls_unlocked or self.mode == self.MODE_SAFE:
                # SAFE: force fixed safe values
                self.steering = self.safe_steering
                self.throttle = self.safe_throttle
                self.gear = self.safe_gear
                self.front_diff = self.safe_front_diff
                self.rear_diff = self.safe_rear_diff
                # camera angle remains whatever (or could also be forced to safe_cam)
            else:
                if self.mode == self.MODE_AUTO:
                    steering_raw, throttle_raw = self.cmd_vel_to_commands(self.last_cmd_vel)
                    steering_robot = steering_raw
                    throttle_robot = throttle_raw

                else:  # MANUAL
                    # Use last joystick axes if available, else keep old commands
                    axes = self.last_joy_axes if self.last_joy_axes else []
                    steering_raw, throttle_raw = self.joy_to_commands(axes)
                    throttle_robot = self.throttle_lookup.cmd_to_speed(throttle_raw)
                    if steering_raw < 0.0:
                        steering_robot = self.neg_angle_lookup.cmd_to_speed(steering_raw)
                    else:
                        steering_robot = self.pos_angle_lookup.cmd_to_speed(steering_raw)

                # Publish steering/throttle after trimming and limits
                real_cmd_msg = Twist()
                real_cmd_msg.linear.x = throttle_robot
                real_cmd_msg.angular.z = steering_robot
                self.pub_real_cmd.publish(real_cmd_msg)

                # Steering directly from raw
                self.steering = clamp(steering_raw, self.min_str, self.max_str)

                # Apply accel limit for throttle
                self.throttle = self.apply_accel_limit(throttle_raw, dt)
                self.throttle = clamp(self.throttle, self.min_vel, self.max_vel)

            # Update camera stabilization
            self.update_camera(dt)

            # Publish control array: [steering, throttle, gear, front diff, rear diff, camera angle]
            msg = Float32MultiArray()
            msg.data = [
                float(self.steering),
                float(self.throttle),
                float(self.gear),
                float(self.front_diff),
                float(self.rear_diff),
                float(self.camera_angle),
            ]
            self.pub_control.publish(msg)

            rate.sleep()


if __name__ == "__main__":
    try:
        VertiWheelControlNode()
    except rospy.ROSInterruptException:
        pass
