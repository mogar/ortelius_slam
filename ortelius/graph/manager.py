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
from dataclasses import dataclass, field

@dataclass
class GraphManagerConfig:
    """Configuration for GraphManager."""
    max_factors_before_update: int = 2
    max_poses_before_marginalization: int = 100
    relinearize_threshold: float = 0.1
    relinearize_skip: int = 1

class GraphManager:
    """Manages the factor graph and ISAM2 optimizer."""
    
    def __init__(self, config: GraphManagerConfig = GraphManagerConfig()):
        self.config = config

        self.isam2_params = ISAM2Params()
        self.isam2_params.setRelinearizeThreshold(self.config.relinearize_threshold)
        self.isam2_params.relinearizeSkip = self.config.relinearize_skip

        self.isam2 = ISAM2(self.isam2_params)

        # Pending additions to the graph before the next update
        self.new_factors = NonlinearFactorGraph()
        self.new_values = Values()

        # Bookkeeping for graph management
        self.pose_keys: list[int] = [] # ordered list of active pose keys
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

    def add_variable(self, key, value) -> None:
        """Add a variable to the graph."""
        self.new_values.insert(key, value)
        if isinstance(value, gtsam.Pose3) or isinstance(value, gtsam.Pose2):
            self.pose_keys.append(key)

    def update_graph(self, force: bool = False) -> None:
        """Trigger an ISAM2 update."""
        self.step_count += 1
        if not force and self.new_factors.size() < self.config.max_factors_before_update:
            return  # Not enough new factors to justify an update

        self.isam2.update(self.new_factors, self.new_values)
        # TODO: extra iterations to relinearize? Maybe check step count instead of just new factor count?

        self.current_estimate = self.isam2.calculateEstimate()

        # Clear the new factors and values after update to prepare for new factors and variables
        self.new_factors.resize(0)
        self.new_values.clear()

        if len(self.pose_keys) > self.config.max_poses_before_marginalization:
            self.marginalize_old_poses()
        
    def marginalize_old_poses(self) -> None:
        """Marginalize old poses to constrain graph size."""
        # GTSAM's marginalizeLeaves removes variables and folds their
        # information into a prior on their neighbors. The key list
        # passed in must be leaves in the Bayes tree — oldest poses
        # usually qualify, but you should verify before generalizing.
        keys_to_remove = gtsam.KeyVector()
        # TODO: check for leaf status before adding to keys_to_remove
        keys_to_remove.append(self.pose_keys[0])

        self.isam2.marginalizeLeaves(keys_to_remove)
        self.pose_keys.pop(0)

    def get_estimate(self, index:int) -> gtsam.Values:
        """Retrieve the current estimate for a variable."""
        key = self.pose_keys[index]
        return self.isam2.calculateEstimate().at(key)

    def get_marginal_covariance(self, index:int) -> ...:
        """Retrieve the marginal covariance for a variable."""
        key = self.pose_keys[index]
        marginals = gtsam.Marginals(self.isam2.getFactorsUnsafe(), self.current_estimate)
        return marginals.marginalCovariance(key)

    def get_all_estimates(self) -> list[tuple[int, gtsam.Value]]:
        """Retrieve the current estimates for all variables."""
        if self.current_estimate is None:
            return []
        return [(key, self.current_estimate.at(key)) for key in self.current_estimate.keys()]
