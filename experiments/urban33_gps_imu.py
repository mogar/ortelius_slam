
"""
urban33_gps_imu.py

Basic experiment with the Ortelius SLAM system, using GPS and IMU data from the Urban33 dataset. This experiment
demonstrates how to set up a factor graph with GPS and IMU factors, and how to run the optimization to estimate the trajectory of the vehicle.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from ortelius.factors.basic_factors import PriorFactor, OdometryFactor2D, GpsUnaryFactor2D
from ortelius.graph.manager import GraphManager, GraphManagerConfig
from ortelius.datasets.kaist_parser import KaistUrbanDataset
import math
from pathlib import Path

import gtsam
import numpy as np


def encoder_to_odometry(prev: EncoderMeasurement, curr: EncoderMeasurement) -> tuple[float, float, float]:
    """Convert two encoder measurements into an odometry measurement (dx, dy, dtheta).

    This function assumes a differential drive robot with two wheel encoders. The encoder counts should be converted
    into metric units (meters and radians) before being used to create the OdometryFactor2D.
    """
    TICKS_PER_REV = 4096
    WHEEL_RADIUS = 0.3  # meters
    WHEEL_BASE = 1.5  # meters
    METERS_PER_TICK = 2 * math.pi * WHEEL_RADIUS / TICKS_PER_REV

    dl = (curr.left_ticks - prev.left_ticks) * METERS_PER_TICK
    dr = (curr.right_ticks - prev.right_ticks) * METERS_PER_TICK

    dtheta = (dr - dl) / WHEEL_BASE
    ds = (dr + dl) / 2
    dx = ds * math.cos(dtheta / 2)
    dy = ds * math.sin(dtheta / 2)
    return dx, dy, dtheta


def gps_sigmas(m: VrsGpsMeasurement) -> tuple[float, float]:
    """Convert GPS measurement accuracy (e.g. HDOP) into standard deviations for the GpsUnaryFactor2D."""
    # lat/long stdev to north/east stdev conversion is not exact, but for small areas we can approximate it as a constant scaling factor.
    scaling = 1.0  # TODO
    sigma_x = m.lat_std * scaling
    sigma_y = m.lon_std * scaling
    print(
        f"GPS measurement at {m.easting}, {m.northing} has lat_std {m.lat_std}, lon_std {m.lon_std}")
    return sigma_x, sigma_y


def make_local_origin(vrs_gps: list[VrsGpsMeasurement]) -> tuple[float, float]:
    """Create a local origin for the factor graph based on the first GPS measurement."""
    # For simplicity, we can just use the first GPS measurement as the local origin. In a real system, you might want to
    # use a more robust method (e.g. averaging multiple measurements) to determine the local origin.
    # TODO: may want to check fix state before selecting the first GPS measurement as the origin
    first_gps = vrs_gps[0]
    return gtsam.Pose2(first_gps.easting, first_gps.northing, 0.0)


def run_experiment(dataset_root: str, max_steps: int = 500):
    dataset = KaistUrbanDataset(Path(dataset_root))

    print(f"GPS measurements: {len(dataset.gps_measurements)}")
    print(f"VRS GPS measurements: {len(dataset.vrs_gps_measurements)}")
    print(f"Encoder measurements: {len(dataset.encoder_measurements)}")
    print(f"IMU measurements: {len(dataset.imu_measurements)}")
    print(f"FOG measurements: {len(dataset.fog_measurements)}")
    print(f"Baseline poses: {len(dataset.baseline_poses)}")

    origin = make_local_origin(dataset.vrs_gps_measurements)
    print(f"Local origin set to: {origin}")

    config = GraphManagerConfig(
        max_factors_before_update=10,
        max_poses_before_marginalization=100,
        relinearize_threshold=0.1,
        relinearize_skip=1,
        lag=0.5,
    )
    graph_manager = GraphManager(config)

    pose_index = 0
    prev_encoder = None
    initialized = False
    step = 0

    for sensor_name, measurement in dataset.iter_synchronized(['encoder', 'vrs']):
        if step >= max_steps:
            break

        m_time = float(measurement.timestamp)/1e9
        if sensor_name == 'vrs':
            # TODO: check fix state? If the GPS measurement is not reliable, we might want to skip adding a factor for it
            gps_x = measurement.easting - origin.x()
            gps_y = measurement.northing - origin.y()
            sigma_x, sigma_y = gps_sigmas(measurement)

            # Note that a GPS measurement doesn't trigger a new pose index, as it's a unary factor on the current pose.
            if not initialized:
                # Add a prior factor for the initial pose based on the first GPS measurement
                key = graph_manager.pose_key(pose_index)
                curr_pose = gtsam.Pose2(gps_x, gps_y, 0.0)
                graph_manager.add_variable(key, curr_pose, m_time)
                for f in PriorFactor(key, gps_x, gps_y, sigma_x, sigma_y).build():
                    graph_manager.add_factor(f)
                initialized = True
            else:
                # TODO: gps factors are breaking things (looking for navstate values during graph update)
                gps_factor = GpsUnaryFactor2D(
                    graph_manager.pose_key(pose_index),
                    gps_x, gps_y,
                    sigma_x, sigma_y
                )
                for f in gps_factor.build():
                    graph_manager.add_factor(f)

        if not initialized:
            continue  # Wait until we have the first GPS measurement to initialize the graph

        if sensor_name == 'encoder':
            if prev_encoder is not None:
                dx, dy, dtheta = encoder_to_odometry(prev_encoder, measurement)

                # skip adding an odometry factor if the movement is too small, to avoid adding noisy factors for very small motions
                if abs(dx) < 1e-4 and abs(dtheta) < 1e-4:
                    continue

                prev_key = graph_manager.pose_key(pose_index)
                prev_pose = graph_manager.get_estimate(pose_index) 
                if prev_pose is None:
                    prev_pose = origin
                pose_index += 1
                curr_key = graph_manager.pose_key(pose_index)
                # dead-reckon the new pose based on odometry
                curr_pose = prev_pose.compose(gtsam.Pose2(dx, dy, dtheta))
                graph_manager.add_variable(curr_key, curr_pose, m_time)

                # TODO: maybe estimate odometry noise based on encoder counts or time delta?
                # For now we use fixed noise parameters
                odom_factor = OdometryFactor2D(
                    prev_key, curr_key,
                    dx, dy, dtheta,
                    sigma_x=0.1, sigma_y=0.1, sigma_theta=0.05
                )
                for f in odom_factor.build():
                    graph_manager.add_factor(f)
            prev_encoder = measurement

        # Update the graph after adding new factors and variables
        graph_manager.update_graph()
        step += 1

        if step % 50 == 0:
            graph_manager.update_graph(True)
            print(f"Step {step}: Added factors and updated graph.")
            current_pose = graph_manager.current_estimate.atPose2(
                graph_manager.pose_key(pose_index))
            print(f"Current estimate for pose {pose_index}: {current_pose}")
            print(
                f"x: {current_pose.x()}, y: {current_pose.y()}, theta: {current_pose.theta()}")

    poses = graph_manager.get_all_estimates()
    print(f"Done. Final trajectory has {len(poses)} poses.")
    return poses


if __name__ == "__main__":
    # For basic example usage, we assume the dataset is already downloaded and extracted to the data subdirectory
    file_path = Path(__file__).resolve().parent
    dataset_path = Path(file_path / '..' / 'ortelius' /
                        'datasets' / 'data' / 'urban33-yeouido')
    trajectory = run_experiment(dataset_path, max_steps=50000)
