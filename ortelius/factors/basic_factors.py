"""
This module defines some basic factors that can be used in a factor graph for SLAM or state estimation.
"""

import gtsam
import numpy as np

from ..graph.factor_base import FactorBase


class PriorFactor(FactorBase):
    """A simple prior factor on a single variable."""
    
    def __init__(self, key: int, value, noise_model):
        self._key = key
        self._value = value
        # TODO: could use a simpler diagonal noise model for priors, but this is flexible for now
        self._noise_model = noise_model

    def build(self) -> list:
        """Build the gtsam PriorFactor."""
        return [gtsam.PriorFactor(self._key, self._value, self._noise_model)]

    def keys(self) -> list[int]:
        """Return the key that this factor depends on."""
        return [self._key]

class OdometryFactor2D(FactorBase):
    """A simple odometry factor between two Pose2 variables.
    
    Encodes relative motion (dx, dy, dtheta) between two poses, typically from odometry or IMU integration.
    If using wheel odometry, encoder counts should be converted into metric units (meters and radians) before
    creating this factor.
    """
    
    def __init__(self, key1: int, key2: int, dx: float, dy: float, dtheta: float, sigma_x: float, sigma_y: float, sigma_theta: float):
        self._key1 = key1
        self._key2 = key2
        self._dx = dx
        self._dy = dy
        self._dtheta = dtheta
        self._relative_pose = gtsam.Pose2(self._dx, self._dy, self._dtheta)
        self._noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_x, sigma_y, sigma_theta]))

    def build(self) -> list:
        """Build the gtsam BetweenFactor for odometry."""
        factor = gtsam.BetweenFactorPose2(
            self._key1,
            self._key2,
            self._relative_pose,
            self._noise_model,
        )
        return [factor]
        
    def keys(self) -> list[int]:
        """Return the keys that this factor depends on."""
        return [self._key1, self._key2]

class GpsUnaryFactor2D(FactorBase):
    """A unary factor that provides a GPS measurement for a Pose2 variable.
    
    Encodes an absolute position measurement (x, y) with some noise. The orientation is not measured by GPS,
    so the factor only constrains the x and y components of the pose.
    """
    
    def __init__(self, key: int, x: float, y: float, sigma_x: float, sigma_y: float):
        self._key = key
        self._x = x
        self._y = y
        self._noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_x, sigma_y]))

    def build(self) -> list:
        """Build the gtsam UnaryFactor for GPS."""
        # We can use a PriorFactor on the Pose2 variable, but only constrain the x and y components.
        # To do this, we create a custom noise model that ignores the theta component.
        factor = gtsam.GPSFactor2(
            self._key,
            [self._x, self._y, 0.0], #
            self._noise_model,
        )
        return [factor]
        
    def keys(self) -> list[int]:
        """Return the key that this factor depends on."""
        return [self._key]