"""Dataset Loader for KAIST Urban Dataset.

Formatting based on description from https://sites.google.com/view/complex-urban-dataset/format?authuser=0
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import csv


# --- Data Types (generic) ---

@dataclass
class GpsMeasurement:
    timestamp: int  # nanoseconds since epoch
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters
    position_covariance: np.ndarray  # 3x3 covariance matrix


@dataclass
class VrsGpsMeasurement:
    timestamp: int  # nanoseconds since epoch
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters
    easting: float  # meters
    northing: float  # meters
    fix_state: int  # 4 = fix, 5 = float, 1 = normal
    lat_std: float
    lon_std: float
    alt_std: float


@dataclass
class EncoderMeasurement:
    timestamp: int  # nanoseconds since epoch
    left_ticks: int
    right_ticks: int


@dataclass
class ImuMeasurement:
    timestamp: int  # nanoseconds since epoch
    quat: np.ndarray  # 4-element array (x, y, z, w)
    euler: np.ndarray  # 3-element array (roll, pitch, yaw)
    gyro: np.ndarray | None  # 3-element array (x, y, z) in rad/s
    accel: np.ndarray | None  # 3-element array (x, y, z) in m/s^2


@dataclass
class FogMeasurement:
    timestamp: int  # nanoseconds since epoch
    delta_roll: float
    delta_pitch: float
    delta_yaw: float


@dataclass
class BaselinePose:
    timestamp: int  # nanoseconds since epoch
    T: np.ndarray  # 4x4 transformation matrix from base_link to map frame


@dataclass
class DataStampEntry:
    timestamp: int  # nanoseconds since epoch
    sensor_name: str  # e.g. "gps", "vrs_gps", etc.

# --- Dataset Loader Helpers ---


def load_gps(path: Path) -> list[GpsMeasurement]:
    """GPS Parser.

    [timestamp, latitude, longitude, altitude, 9-tuple vector (position covariance)]
    """
    measurements = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            timestamp = int(row[0])
            latitude = float(row[1])
            longitude = float(row[2])
            altitude = float(row[3])
            cov_flat = [float(x) for x in row[4:13]]
            position_covariance = np.array(cov_flat).reshape(3, 3)
            measurements.append(GpsMeasurement(
                timestamp, latitude, longitude, altitude, position_covariance))
    return measurements


def load_vrs_gps(path: Path) -> list[VrsGpsMeasurement]:
    """VRS GPS parser. Note that the VRS GPS data has a different format than the regular GPS data, so we need a separate parser. 

    VRS GPS has two types, but the fields we are interested in are in the same columns for both types so we ignore that.

    Ver 1: [timestamp, 
            latitude, longitude, 
            x coordinate, y coordinate, 
            altitude, fix state, number of satellite, horizontal precision, 
            latitude std, longitude std, altitude std, 
            heading validate flag , magnetic global heading, 
            speed in knot, speed in km, GNVTG mode] 
    """
    measurements = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            timestamp = int(row[0])
            latitude = float(row[1])
            longitude = float(row[2])
            altitude = float(row[5])
            easting = float(row[3])
            northing = float(row[4])
            fix_state = int(row[6])
            lat_std = float(row[9])
            lon_std = float(row[10])
            alt_std = float(row[11])
            measurements.append(VrsGpsMeasurement(timestamp, latitude, longitude,
                                altitude, easting, northing, fix_state, lat_std, lon_std, alt_std))
    return measurements


def load_encoder(path: Path) -> list[EncoderMeasurement]:
    """Encoder parser.

    [timestamp, left count, right count]
    """
    measurements = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            timestamp = int(row[0])
            left_ticks = int(row[1])
            right_ticks = int(row[2])
            measurements.append(EncoderMeasurement(
                timestamp, left_ticks, right_ticks))
    return measurements


def load_imu(path: Path) -> list[ImuMeasurement]:
    """IMU Parser.

    [timestamp, 
     quaternion x, quaternion y, quaternion z, quaternion w, 
     Euler x, Euler y, Euler z, 
     Gyro x, Gyro y, Gyro z, 
     Acceleration x, Acceleration y, Acceleration z, 
     MagnetField x, MagnetField y, MagnetField z]
    """
    measurements = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            timestamp = int(row[0])
            quat = np.array([float(row[1]), float(row[2]),
                             float(row[3]), float(row[4])])
            euler = np.array([float(row[5]), float(row[6]), float(row[7])])
            gyro = accel = None
            if len(row) >= 14:   # Ver2
                gyro = np.array(
                    [float(row[8]),  float(row[9]),  float(row[10])])
                accel = np.array(
                    [float(row[11]), float(row[12]), float(row[13])])
            measurements.append(ImuMeasurement(
                timestamp, quat, euler, gyro, accel))
    return measurements


def load_fog(path: Path) -> list[FogMeasurement]:
    """Fog parser.

    [timestamp, delta roll, delta pitch, delta yaw]
    """
    measurements = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            timestamp = int(row[0])
            delta_roll = float(row[1])
            delta_pitch = float(row[2])
            delta_yaw = float(row[3])
            measurements.append(FogMeasurement(
                timestamp, delta_roll, delta_pitch, delta_yaw))
    return measurements


def load_baseline_poses(path: Path) -> list[BaselinePose]:
    """Baseline pose parser.

    Baseline is not suggested for use as ground truth comparison with SLAM algorithms.

    [timestamp, P(0,0), P(0,1), P(0,2), P(0,3), P(1,0), P(1,1), P(1,2), P(1,3), P(2,0), P(2,1), P(2,2), P(2,3)] 
    """
    poses = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            timestamp = int(row[0])
            vals = [float(x) for x in row[1:13]]
            # Row-major 3x4, pad to 4x4
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(BaselinePose(timestamp, T))
    return poses


def load_data_stamps(path: Path) -> list[DataStampEntry]:
    """Data stamp parser.

    [timestamp, sensor name]
    """
    entries = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            timestamp = int(row[0])
            sensor_name = row[1].strip()
            entries.append(DataStampEntry(timestamp, sensor_name))
    return entries

# --- Dataset Loader Class ---


class KaistUrbanDataset:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        sensor_path = root_path / 'sensor_data'
        self.gps_measurements = load_gps(sensor_path / 'gps.csv')
        self.vrs_gps_measurements = load_vrs_gps(sensor_path / 'vrs_gps.csv')
        self.encoder_measurements = load_encoder(sensor_path / 'encoder.csv')
        self.imu_measurements = load_imu(sensor_path / 'xsens_imu.csv')
        self.fog_measurements = load_fog(sensor_path / 'fog.csv')
        self.data_stamps = load_data_stamps(sensor_path / 'data_stamp.csv')

        self.baseline_poses = load_baseline_poses(
            root_path / 'global_pose.csv')

        self._build_index()

    def _build_index(self):
        # Build timestamp index for each sensor type for fast lookup
        self.gps_index = {m.timestamp: m for m in self.gps_measurements}
        self.vrs_gps_index = {
            m.timestamp: m for m in self.vrs_gps_measurements}
        self.encoder_index = {
            m.timestamp: m for m in self.encoder_measurements}
        self.imu_index = {m.timestamp: m for m in self.imu_measurements}
        self.fog_index = {m.timestamp: m for m in self.fog_measurements}
        self.baseline_pose_index = {
            p.timestamp: p for p in self.baseline_poses}

    def iter_synchronized(self, sensors: list[str]):
        """
        Yield measurements in timestamp order for the requested sensor types.
        sensors: subset of ['encoder', 'imu', 'vrs', 'fog', ...]
                 as they appear in data_stamp.csv
        """
        sensor_set = set(sensors)
        lookup = {
            'encoder': self.encoder_index,
            'imu':     self.imu_index,
            'vrs':     self.vrs_gps_index,
            'gps':     self.gps_index,
            'fog':     self.fog_index,
        }
        for entry in self.data_stamps:
            if entry.sensor_name in sensor_set:
                table = lookup.get(entry.sensor_name)
                if table and entry.timestamp in table:
                    # TODO: dealing with images, lidar
                    yield entry.sensor_name, table[entry.timestamp]


# --- Example usage ---

if __name__ == "__main__":
    # For basic example usage, we assume the dataset is already downloaded and extracted to the data subdirectory
    file_path = Path(__file__).resolve().parent
    dataset = KaistUrbanDataset(Path(file_path / 'data' / 'urban33-yeouido'))

    print(f"GPS measurements: {len(dataset.gps_measurements)}")
    print(f"VRS GPS measurements: {len(dataset.vrs_gps_measurements)}")
    print(f"Encoder measurements: {len(dataset.encoder_measurements)}")
    print(f"IMU measurements: {len(dataset.imu_measurements)}")
    print(f"FOG measurements: {len(dataset.fog_measurements)}")
    print(f"Baseline poses: {len(dataset.baseline_poses)}")

    loop_count = 0
    loop_limit = 100
    for sensor_name, measurement in dataset.iter_synchronized(['encoder', 'imu', 'gps']):
        print(f"{loop_count}:{sensor_name} @ {measurement.timestamp}: {measurement}")
        loop_count += 1
        if loop_count >= loop_limit:
            break
