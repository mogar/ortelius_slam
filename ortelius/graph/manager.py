"""Graph Manager.

Handles graph setup and bookkeeping. Also acts as a wrapper to GTSAM operations.

Major responsibilities include:

* variable management
* factor accumulation
* triggering graph updates (isam2.update())
* retrieving estimates from the graph
* marginalization to remove old poses and constrain graph size
"""

# TODO: how to handle node type (Pose3 vs Pose2) in a more flexible way? Maybe a template parameter or separate classes?

import gtsam
from gtsam import ISAM2, ISAM2Params, NonlinearFactorGraph, Values
from gtsam_unstable import IncrementalFixedLagSmoother, FixedLagSmootherKeyTimestampMap
from dataclasses import dataclass, field


@dataclass
class GraphManagerConfig:
    """Configuration for GraphManager."""
    max_factors_before_update: int = 2
    max_poses_before_marginalization: int = 100
    relinearize_threshold: float = 0.1
    relinearize_skip: int = 1
    lag: float = 30.0


class GraphManager:
    """Manages the factor graph and ISAM2 optimizer."""

    def __init__(self, config: GraphManagerConfig = GraphManagerConfig()):
        self.config = config

        self.isam2_params = ISAM2Params()
        self.isam2_params.setRelinearizeThreshold(
            self.config.relinearize_threshold)
        self.isam2_params.relinearizeSkip = self.config.relinearize_skip

        self.graph = IncrementalFixedLagSmoother(self.config.lag)

        # Pending additions to the graph before the next update
        self.new_factors = NonlinearFactorGraph()
        self.new_values = Values()

        # Bookkeeping for graph management
        self.pose_keys: list[int] = []  # ordered list of active pose keys
        self.new_timestamps = FixedLagSmootherKeyTimestampMap()
        self.step_count = 0
        self.current_estimate: Values | None = None

    # --- Key Management ---

    def pose_key(self, index: int) -> int:
        """Generate a unique key for a pose variable."""
        return gtsam.symbol('x', index)

    def landmark_key(self, index: int) -> int:
        """Generate a unique key for a landmark variable."""
        return gtsam.symbol('l', index)

    def calib_key(self, index: int) -> int:
        """Generate a unique key for a calibration variable."""
        return gtsam.symbol('c', index)

    # --- Graph Operations ---

    def add_factor(self, factor) -> None:
        """Add a factor to the graph."""
        self.new_factors.add(factor)

    def add_variable(self, key, value, timestamp) -> None:
        """Add a variable to the graph."""
        self.new_values.insert(key, value)
        self.new_timestamps.insert((key, timestamp))
        if isinstance(value, gtsam.Pose3) or isinstance(value, gtsam.Pose2):
            self.pose_keys.append(key)

    def update_graph(self, force: bool = False) -> None:
        """Trigger a graph update."""
        self.step_count += 1
        if not force and self.new_factors.size() < self.config.max_factors_before_update:
            return  # Not enough new factors to justify an update

        # Note that with a fixed lag smoother the old values get marginalized out automatically
        self.graph.update(self.new_factors, self.new_values, self.new_timestamps)
        # TODO: extra iterations to relinearize? Maybe check step count instead of just new factor count?

        self.current_estimate = self.graph.calculateEstimate()

        # Clear the new factors and values after update to prepare for new factors and variables
        self.new_factors.resize(0)
        self.new_values.clear()
        self.new_timestamps.clear()

    def get_estimate(self, index: int) -> gtsam.Values:
        """Retrieve the current estimate for a variable."""
        key = self.pose_keys[index]
        if self.current_estimate is None or not self.current_estimate.exists(key):
            if self.new_values.exists(key):
                return self.new_values.atPose2(key)
            return None
        return self.current_estimate.atPose2(key)

    def get_marginal_covariance(self, index: int) -> ...:
        """Retrieve the marginal covariance for a variable."""
        key = self.pose_keys[index]
        marginals = gtsam.Marginals(
            self.graph.getFactorsUnsafe(), self.current_estimate)
        return marginals.marginalCovariance(key)

    def get_all_estimates(self) -> list[tuple[int, gtsam.Value]]:
        """Retrieve the current estimates for all variables."""
        if self.current_estimate is None:
            return []
        return [(key, self.current_estimate.at(key)) for key in self.current_estimate.keys()]
