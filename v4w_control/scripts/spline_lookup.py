#!/usr/bin/env python3

"""
Spline Lookup Table Module

This module provides a class for creating and using lookup tables from spline calibration data.
It supports both forward (command -> speed) and inverse (speed -> command) lookups with 
high-resolution interpolation.

Usage:
    # Create lookup table from existing spline
    lookup = SplineLookupTable(spline, resolution=1000)
    
    # Forward lookup: command to speed
    speed = lookup.cmd_to_speed(0.5)
    
    # Inverse lookup: speed to command  
    command = lookup.speed_to_cmd(1.2)
    
    # Save/load lookup tables
    lookup.save('my_lookup_table')
    lookup = SplineLookupTable.load('my_lookup_table')
"""

import numpy as np
import pickle
import json
import os
from scipy.interpolate import interp1d, UnivariateSpline
from typing import Union, Tuple, Dict, Any
import warnings


class SplineLookupTable:
    """
    A lookup table class for spline interpolation with forward and inverse capabilities.
    """
    
    def __init__(self, spline: UnivariateSpline = None, cmd_range: Tuple[float, float] = (0.0, 1.0), 
                 resolution: int = 1000, name: str = "spline_lookup"):
        """
        Initialize the lookup table.
        
        Args:
            spline: scipy UnivariateSpline object
            cmd_range: Tuple of (min_cmd, max_cmd) for the lookup range
            resolution: Number of points in the lookup table
            name: Name identifier for the lookup table
        """
        self.name = name
        self.resolution = resolution
        self.cmd_range = cmd_range
        self.speed_range = None
        
        # Lookup functions
        self.forward_lookup = None
        self.inverse_lookup = None
        
        # Data arrays
        self.cmd_values = None
        self.speed_values = None
        self.speed_unique = None
        self.cmd_unique = None
        
        # Metadata
        self.metadata = {}
        
        if spline is not None:
            self._build_from_spline(spline)
    
    def _build_from_spline(self, spline: UnivariateSpline):
        """Build lookup tables from a UnivariateSpline object."""
        # Create high-resolution lookup arrays
        self.cmd_values = np.linspace(self.cmd_range[0], self.cmd_range[1], self.resolution)
        self.speed_values = spline(self.cmd_values)
        
        # Set speed range
        self.speed_range = (float(self.speed_values.min()), float(self.speed_values.max()))
        
        # Create forward lookup table (cmd -> speed)
        self.forward_lookup = interp1d(
            self.cmd_values, self.speed_values, 
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # Create inverse lookup table (speed -> cmd)
        # Remove duplicate speeds to avoid issues with inverse interpolation
        unique_mask = np.concatenate([[True], np.diff(self.speed_values) > 1e-10])
        self.speed_unique = self.speed_values[unique_mask]
        self.cmd_unique = self.cmd_values[unique_mask]
        
        self.inverse_lookup = interp1d(
            self.speed_unique, self.cmd_unique,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # Store metadata
        self.metadata = {
            'spline_type': 'UnivariateSpline',
            'spline_degree': spline.get_knots().shape[0] - 1 if hasattr(spline, 'get_knots') else 'unknown',
            'cmd_range': self.cmd_range,
            'speed_range': self.speed_range,
            'resolution': self.resolution,
            'unique_points': len(self.speed_unique)
        }
    
    def cmd_to_speed(self, cmd_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert velocity command to speed using forward lookup table.
        
        Args:
            cmd_val: Velocity command value(s)
            
        Returns:
            Speed value(s) in m/s
        """
        if self.forward_lookup is None:
            raise ValueError("Lookup table not initialized. Create from spline first.")
        
        result = self.forward_lookup(cmd_val)
        
        # Return scalar if input was scalar
        if np.isscalar(cmd_val):
            return float(result)
        return result
    
    def speed_to_cmd(self, speed_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert speed to velocity command using inverse lookup table.
        
        Args:
            speed_val: Speed value(s) in m/s
            
        Returns:
            Velocity command value(s)
        """
        if self.inverse_lookup is None:
            raise ValueError("Lookup table not initialized. Create from spline first.")
        
        result = self.inverse_lookup(speed_val)
        
        # Return scalar if input was scalar
        if np.isscalar(speed_val):
            return float(result)
        return result
    
    def get_valid_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get the valid ranges for commands and speeds."""
        return {
            'cmd_range': self.cmd_range,
            'speed_range': self.speed_range
        }
    
    def test_accuracy(self, n_points: int = 100) -> Dict[str, float]:
        """
        Test the round-trip accuracy of the lookup tables.
        
        Args:
            n_points: Number of test points
            
        Returns:
            Dictionary with accuracy metrics
        """
        if self.forward_lookup is None or self.inverse_lookup is None:
            raise ValueError("Lookup tables not initialized.")
        
        # Test forward then inverse
        test_cmds = np.linspace(self.cmd_range[0], self.cmd_range[1], n_points)
        test_speeds = self.cmd_to_speed(test_cmds)
        cmds_roundtrip = self.speed_to_cmd(test_speeds)
        
        # Calculate errors
        cmd_errors = np.abs(test_cmds - cmds_roundtrip)
        
        # Test inverse then forward  
        test_speeds_inv = np.linspace(self.speed_range[0], self.speed_range[1], n_points)
        test_cmds_inv = self.speed_to_cmd(test_speeds_inv)
        speeds_roundtrip = self.cmd_to_speed(test_cmds_inv)
        
        speed_errors = np.abs(test_speeds_inv - speeds_roundtrip)
        
        return {
            'max_cmd_error': float(np.max(cmd_errors)),
            'mean_cmd_error': float(np.mean(cmd_errors)),
            'max_speed_error': float(np.max(speed_errors)),
            'mean_speed_error': float(np.mean(speed_errors)),
            'cmd_std_error': float(np.std(cmd_errors)),
            'speed_std_error': float(np.std(speed_errors))
        }
    
    def save(self, filename: str, save_json: bool = True, save_pickle: bool = True):
        """
        Save the lookup table to files.
        
        Args:
            filename: Base filename (without extension)
            save_json: Whether to save human-readable JSON file
            save_pickle: Whether to save binary pickle file for fast loading
        """
        if self.forward_lookup is None or self.inverse_lookup is None:
            raise ValueError("Lookup tables not initialized.")
        
        # Prepare data for saving
        save_data = {
            'name': self.name,
            'resolution': self.resolution,
            'cmd_range': self.cmd_range,
            'speed_range': self.speed_range,
            'metadata': self.metadata,
            'forward_table': {
                'cmd_values': self.cmd_values.tolist(),
                'speed_values': self.speed_values.tolist()
            },
            'inverse_table': {
                'speed_values': self.speed_unique.tolist(),
                'cmd_values': self.cmd_unique.tolist()
            }
        }
        
        # Save JSON file
        if save_json:
            json_file = f"{filename}.json"
            with open(json_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"Saved JSON lookup table: {json_file}")
        
        # Save pickle file with interpolation functions
        if save_pickle:
            pickle_file = f"{filename}.pkl"
            pickle_data = {
                'forward_lookup': self.forward_lookup,
                'inverse_lookup': self.inverse_lookup,
                'lookup_table': self
            }
            with open(pickle_file, 'wb') as f:
                pickle.dump(pickle_data, f)
            print(f"Saved pickle lookup table: {pickle_file}")
    
    @classmethod
    def load(cls, filename: str) -> 'SplineLookupTable':
        """
        Load a lookup table from file.
        
        Args:
            filename: Base filename (tries .pkl first, then .json)
            
        Returns:
            SplineLookupTable instance
        """
        # Try pickle file first (faster)
        pickle_file = f"{filename}.pkl"
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
                if 'lookup_table' in data:
                    return data['lookup_table']
                else:
                    # Legacy format - reconstruct
                    return cls._load_from_pickle_legacy(data)
        
        # Try JSON file
        json_file = f"{filename}.json"
        if os.path.exists(json_file):
            return cls._load_from_json(json_file)
        
        raise FileNotFoundError(f"Could not find lookup table files: {pickle_file} or {json_file}")
    
    @classmethod
    def _load_from_json(cls, json_file: str) -> 'SplineLookupTable':
        """Load lookup table from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create instance
        instance = cls(
            cmd_range=tuple(data['cmd_range']),
            resolution=data['resolution'],
            name=data.get('name', 'loaded_lookup')
        )
        
        # Load data arrays
        instance.cmd_values = np.array(data['forward_table']['cmd_values'])
        instance.speed_values = np.array(data['forward_table']['speed_values'])
        instance.speed_unique = np.array(data['inverse_table']['speed_values'])
        instance.cmd_unique = np.array(data['inverse_table']['cmd_values'])
        
        # Set ranges
        instance.speed_range = tuple(data['speed_range'])
        instance.metadata = data.get('metadata', {})
        
        # Recreate interpolation functions
        instance.forward_lookup = interp1d(
            instance.cmd_values, instance.speed_values,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        instance.inverse_lookup = interp1d(
            instance.speed_unique, instance.cmd_unique,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        return instance
    
    @classmethod
    def _load_from_pickle_legacy(cls, data: dict) -> 'SplineLookupTable':
        """Load from legacy pickle format."""
        instance = cls()
        instance.forward_lookup = data['forward_lookup']
        instance.inverse_lookup = data['inverse_lookup']
        
        # Try to extract metadata if available
        if 'metadata' in data:
            metadata = data['metadata']
            instance.metadata = metadata
            instance.cmd_range = tuple(metadata.get('cmd_range', (0.0, 1.0)))
            instance.speed_range = tuple(metadata.get('speed_range', (0.0, 2.0)))
            instance.resolution = metadata.get('resolution', 1000)
        
        return instance
    
    @classmethod
    def from_data_points(cls, cmd_points: np.ndarray, speed_points: np.ndarray, 
                        spline_degree: int = 2, smoothing: float = 0.0, 
                        cmd_range: Tuple[float, float] = None,
                        resolution: int = 1000, name: str = "data_lookup") -> 'SplineLookupTable':
        """
        Create lookup table from raw data points by fitting a spline.
        
        Args:
            cmd_points: Array of command values
            speed_points: Array of corresponding speed values
            spline_degree: Degree of spline (1=linear, 2=quadratic, 3=cubic)
            smoothing: Smoothing factor for spline fitting
            cmd_range: Range for lookup table (defaults to data range)
            resolution: Number of points in lookup table
            name: Name for the lookup table
            
        Returns:
            SplineLookupTable instance
        """
        # Fit spline to data
        spline = UnivariateSpline(cmd_points, speed_points, k=spline_degree, s=smoothing)
        
        # Set range if not provided
        if cmd_range is None:
            cmd_range = (float(np.min(cmd_points)), float(np.max(cmd_points)))
        
        # Create lookup table
        return cls(spline=spline, cmd_range=cmd_range, resolution=resolution, name=name)
    
    def plot(self, show_data_points: bool = False, data_cmd: np.ndarray = None, 
             data_speed: np.ndarray = None):
        """
        Plot the lookup tables for visualization.
        
        Args:
            show_data_points: Whether to show original data points
            data_cmd: Original command data points
            data_speed: Original speed data points
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        if self.forward_lookup is None or self.inverse_lookup is None:
            raise ValueError("Lookup tables not initialized.")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Forward lookup
        axes[0].plot(self.cmd_values, self.speed_values, 'b-', linewidth=2, label='Lookup Table')
        if show_data_points and data_cmd is not None and data_speed is not None:
            axes[0].plot(data_cmd, data_speed, 'ro', markersize=8, label='Original Data')
        axes[0].set_xlabel('Velocity Command')
        axes[0].set_ylabel('Speed (m/s)')
        axes[0].set_title('Forward Lookup: Command → Speed')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Inverse lookup
        axes[1].plot(self.speed_unique, self.cmd_unique, 'g-', linewidth=2, label='Inverse Lookup')
        if show_data_points and data_cmd is not None and data_speed is not None:
            axes[1].plot(data_speed, data_cmd, 'ro', markersize=8, label='Original Data')
        axes[1].set_xlabel('Speed (m/s)')
        axes[1].set_ylabel('Velocity Command')
        axes[1].set_title('Inverse Lookup: Speed → Command')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Round-trip error
        accuracy = self.test_accuracy()
        test_cmds = np.linspace(self.cmd_range[0], self.cmd_range[1], 100)
        test_speeds = self.cmd_to_speed(test_cmds)
        cmd_roundtrip = self.speed_to_cmd(test_speeds)
        errors = np.abs(test_cmds - cmd_roundtrip)
        
        axes[2].plot(test_cmds, errors, 'r-', linewidth=2)
        axes[2].set_xlabel('Original Command')
        axes[2].set_ylabel('Round-trip Error')
        axes[2].set_title(f'Round-trip Error\nMax: {accuracy["max_cmd_error"]:.2e}')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """String representation of the lookup table."""
        if self.forward_lookup is None:
            return f"SplineLookupTable('{self.name}') - Not initialized"
        
        return (f"SplineLookupTable('{self.name}')\n"
                f"  Command range: {self.cmd_range}\n"
                f"  Speed range: {self.speed_range}\n"
                f"  Resolution: {self.resolution}\n"
                f"  Unique points: {len(self.speed_unique) if self.speed_unique is not None else 'N/A'}")