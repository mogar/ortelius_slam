from abc import ABC, abstractmethod
import gtsam

class FactorBase(ABC):
    """Abstract base class for factors in the factor graph.
    
    Subclasses should implement the build() method to return a list of gtsam factors,
    and the keys() method to return the list of variable keys that this factor depends on.

    In general, the graph manager will call build() to get the gtsam factors to add to the graph, 
    and keys() to know which variables are involved for bookkeeping and marginalization purposes.
    """

    @abstractmethod
    def build(self) -> list:
        """Return a list of gtsam factors that this factor represents.
        
        Most factors return a single element list. Multiple factors are supported because some 
        complex factors (e.g. a GPS factor that constrains both position and velocity) might 
        need to return multiple gtsam factors to represent the full measurement.
        """
        ...

    @abstractmethod
    def keys(self) -> list[int]:
        """Return a list of keys that this factor depends on."""
        ...