#!/usr/bin/env python3

import numpy as np
import torch
import tf

from functools import lru_cache
import rospy
from sensor_msgs.msg import Imu, MagneticField, Joy
from std_msgs.msg import Float32MultiArray
from ahrs.filters import Madgwick, Mahony, Complementary, EKF
from scipy.fft import fft, ifft
from functools import lru_cache
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
import matplotlib.pyplot as plt

class ImuProcessor:
    '''Class to process IMU data and estimate the orientation of the vehicle
    using different algorithms. The class is designed to be used with ROS based on AHRS library.

    Parameters
    ----------
    imu_topic : str, default: '/imu/data'
        Topic where the IMU data is published
    mag_topic: str, default: None
        Topic where the magnetometer data is published
    use_mag: bool, default: True
        Flag to enable/disable magnetometer integration
    sampling_rate: int, default: 200
        Sampling frequency in Herz.
    init_time: int, default: 2
        Time in seconds to initialize the orientation estimation quaternion
    method: str, default: 'madgwick'
        Method to estimate the orientation of the vehicle. Supported methods are:
        madgwick, mahony, complementary, ekf
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).
    initial_q: np.array, default: None
        Initial orientation as a quaternion
    initialized: bool, default: False
        Flag to indicate if the orientation has been initialized
    vel_cutoff: int, default: None
        Cutoff frequency for the velocity FFT filter
    vel_deadzone: float, default: None
        Deadzone for the velocity filter

    Attributes
    ----------
    ahrs : object
        AHRS filter object
    method : str
        Method used to estimate the orientation
    initialized : bool
        Flag to indicate if the orientation has been initialized
    init_frames : int
        Number of frames to initialize the orientation
    Q : np.array
        Orientation quaternion
    gyro_cache : np.array
        Cache to store the gyroscope data 
    imu_data : list
        List to store the IMU data
    mag_data : list
        List to store the magnetometer data
    angles : np.array
        Euler angles of the orientation
    velocities : np.array
        Velocities of the vehicle
    
    Raises
    ------
    ValueError
        If the method is not supported
    
    Examples
    --------
    Assuming that the IMU 9DoF running ROS driver node and IMU data is published in the topic /imu/data and the magnetometer data is published in the topic /mag/data
    IMU data is sampled at 200Hz and the orientation is estimated using the Madgwick algorithm
    
    >>> from callbacks import imu_processor
    >>> imu = imu_processor(imu_topic='/imu/data', mag_topic='/mag/data', use_mag=True, sampling_rate=200, init_time=2, method='madgwick', initial_q=None, initialized=False, vel_cutoff=None, vel_deadzone=None)
    >>> roll, pitch, yaw = imu.angles
    >>> vRoll, vPitch, vYaw = imu.velocities
    '''

    def __init__(self, imu_topic=None, mag_topic=None, use_mag=True, sampling_rate=None, 
                 init_time = 2, method='madgwick', initial_q=None, initialized=False, 
                 vel_cutoff=None, vel_deadzone=None, frame='NED', tait_bryan=False):    
        
        '''Initialize suscribers to the IMU and magnetometer topics'''
        print("--------IMU processor with parameters: --------\n")
        if imu_topic is None:
            print("IMU topic not provided. Defaulting to \t /imu/data")
            imu_topic = '/imu/data'
        else:
            print(f"Suscribed to IMU at topic \t\t {imu_topic}")

        rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=800)   
        
        self.use_mag = False if mag_topic is None else use_mag
        if self.use_mag:
            print(f"Suscribed to magnetometer at topic \t {mag_topic}")
            rospy.Subscriber(mag_topic, MagneticField, self.magnetometer_callback, queue_size=800)
        else:
            print("Magnetometer not used")

        if sampling_rate is None:
            print("Sampling rate not provided. Defaulting to \t 200Hz")
            self.sampling_rate = 200
        else:
            print(f"Sampling rate set \t\t\t {sampling_rate}Hz")
            self.sampling_rate = sampling_rate

        if vel_cutoff is None:
            self.cutoff_frequency = int(sampling_rate / 8)
            print(f"Velocity filter Cutoff frequency not provided. Defaulting to \t{self.cutoff_frequency}Hz")
        else:
            print(f"Velocity filter Cutoff frequency set \t{vel_cutoff}Hz")
            self.cutoff_frequency = vel_cutoff

        if vel_deadzone is None:
            self.vel_deadzone = 0.02
            print(f"velocity Deadzone not provided. Defaulting to \t{self.vel_deadzone}")
        else:
            print(f"velocity Deadzone set to \t{vel_deadzone}")
            self.vel_deadzone = vel_deadzone

        self.frame = frame

        if method == 'madgwick':
            self.ahrs = Madgwick(frequency=self.sampling_rate, gain=0.038)
        elif method == 'mahony':
            self.ahrs = Mahony(frequency=self.sampling_rate)
        elif method == 'complementary':
            self.ahrs = Complementary(frequency=self.sampling_rate)
        elif method == 'ekf':
            self.ahrs = EKF(frequency=self.sampling_rate, frame=self.frame, noises=[0.2**2, 0.4**2, 0.95**2])
            print(f"Using EKF with frame \t\t\t {frame}")
        else:
            raise ValueError(f"Method {method} not supported. Supported methods are: madgwick, mahony, complementary, ekf")
        
        self.method = method
        self.tait_bryan = tait_bryan

        print(f"Estimation method set to \t\t\t {method}")

        self.initialized = initialized
        self.init_frames = init_time * self.sampling_rate
        print(f"Initialization time set to \t\t {init_time}s")
        print("----------------------------------------------------")
        print("Initializing orientation...")

        self.Q = initial_q
        self.gyro_cache = np.zeros((self.sampling_rate, 3), dtype=np.float32)
        self.mag_cache = np.zeros((self.sampling_rate, 3), dtype=np.float32)
        self.acc_cache = np.zeros((self.sampling_rate, 3), dtype=np.float32)

        self.imu_data = []
        if self.use_mag:
            self.mag_data = []
        else:
            self.mag_data = None
        
        self.angles = np.zeros(3, dtype=np.float32)
        self.velocities = np.zeros(3, dtype=np.float32)

    def initialize_q(self):
        if self.mag_data:
            if len(self.mag_data) < len(self.imu_data):
                return
            self.ahrs.mag = np.array(self.mag_data, dtype=np.float32)
        
        dataset = np.array(self.imu_data, dtype=np.float32)
        self.ahrs.acc = dataset[:, 0]
        self.ahrs.gyr = dataset[:, 1]

        if self.method == 'ekf':
            self.Q = self.ahrs._compute_all(frame=self.frame)[-1]
        else:
            self.Q = self.ahrs._compute_all()[-1]
        
        self.velocities = self.apply_fft_filter(dataset[:, 1])[-1]
        if self.tait_bryan:
            self.angles = np.array(quaternion_to_tait_bryan_angles(self.Q), dtype=np.float32)
        else:
            q = Quaternion()
            q.w, q.x, q.y, q.z = self.Q
            self.angles = np.array(quaternion_to_angle(q), dtype=np.float32)

        self.initialized = True
        print(f"Orientation initialized to\t roll: {self.angles[0]:.2f},\t pitch: {self.angles[1]:.2f},\t yaw: {self.angles[2]:.2f}")
        print(f"Velocities initialized to\t vroll: {self.velocities[0]:.2f},\t vpitch:{self.velocities[1]:.2f},\t vyaw: {self.velocities[2]:.2f}")
        self.imu_data = np.array([dataset[-1, 0], dataset[-1, 1]] , dtype=np.float32)
        self.mag_data = self.mag_data[-1] if not self.mag_data is None else None

    def imu_callback(self, msg):
        acc_data = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=np.float32)

        gyro_data = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float32)

        if not self.initialized:
            if len(self.imu_data) < self.init_frames:
                self.imu_data.append([acc_data, gyro_data])
            
            else:
                self.imu_data.append([acc_data, gyro_data])
                self.imu_data.pop(0)
                self.initialize_q()
            return
        
        self.imu_data = np.array([acc_data, gyro_data], dtype=np.float32)
        
        self.acc_cache = np.roll(self.acc_cache, -1, axis=0)
        self.acc_cache[-1] = acc_data
        
        self.gyro_cache = np.roll(self.gyro_cache, -1, axis=0)
        self.gyro_cache[-1] = gyro_data

        self.update_orientation()

    def magnetometer_callback(self, msg):
        magField = np.array([
            msg.magnetic_field.x,
            msg.magnetic_field.y,
            msg.magnetic_field.z
        ], dtype=np.float32)

        self.mag_cache = np.roll(self.mag_cache, -1, axis=0)
        self.mag_cache[-1] = magField

        if not self.initialized:
            if len(self.mag_data) < self.init_frames:
                self.mag_data.append(magField)
            return
    
        self.mag_data = magField
    
    def plot_graph(self, data1, data2, fig, ax):
        
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()
        ax[0].plot(data1[:, 0], 'r', label='Filtered')
        ax[0].plot(data2[:, 1], 'b', label='Raw')
        ax[0].set_title('X-axis')
        ax[1].plot(data1[:, 1], 'r', label='Filtered')
        ax[1].plot(data2[:, 1], 'b', label='Raw')
        ax[1].set_title('Y-axis')
        ax[2].plot(data1[:, 2], 'r', label='Filtered')
        ax[2].plot(data2[:, 2], 'b', label='Raw')
        ax[2].set_title('Z-axis')
        fig.canvas.draw()

    def update_orientation(self):
        if self.imu_data is None:
            return

        accel = self.imu_data[0]
        gyro  = self.imu_data[1]
        magField = self.mag_data

        accel = self.apply_fft_filter(self.acc_cache)[-1]
        gyro  = self.imu_data[1]
        magField = self.apply_fft_filter(self.mag_cache)[-1]

        if self.method == 'madgwick':
            self.Q = self.ahrs.updateMARG(q=self.Q, gyr=gyro, acc=accel, mag=magField)
        
        elif self.method == 'mahony':
            if self.use_mag:
                self.Q = self.ahrs.updateMARG(q=self.Q, gyr=gyro, acc=accel, mag=magField)
            else:
                self.Q = self.ahrs.updateIMU(q=self.Q, gyr=gyro, acc=accel)
        
        elif self.method == 'complementary':
            self.Q = self.ahrs.update(q=self.Q, gyr=gyro, acc=accel, mag=magField)

        elif self.method == 'ekf':
            self.Q = self.ahrs.update(q=self.Q, gyr=gyro, acc=accel, mag=magField)
        
        if self.tait_bryan:
            self.angles = np.array(quaternion_to_tait_bryan_angles(self.Q), dtype=np.float32)
        else:
            q = Quaternion()
            q.w, q.x, q.y, q.z = self.Q
            self.angles = np.array(quaternion_to_angle(q), dtype=np.float32)
        
        # self.angles = np.degrees(self.angles)
        '''update velocities using gyro data and FFT'''
        self.velocities = self.apply_fft_filter(self.gyro_cache)[-1]
        self.velocities[np.abs(self.velocities) < self.vel_deadzone] = 0.0

    def apply_fft_filter(self, data):
        """
        Apply a low-pass filter using FFT to the gyroscope data.
        """
        # Perform FFT along the time axis
        fft_coeffs = fft(data, axis=0)

        # Calculate frequencies
        freqs = np.fft.fftfreq(data.shape[0], d=1.0 / self.sampling_rate)

        # Create low-pass filter mask
        filter_mask = np.abs(freqs) <= self.cutoff_frequency

        # Apply filter
        filtered_fft_coeffs = fft_coeffs * filter_mask[:, np.newaxis]

        # Inverse FFT to transform back to time domain
        filtered_data = np.real(ifft(filtered_fft_coeffs, axis=0))

        return filtered_data

