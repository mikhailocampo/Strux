from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from strux.types import MetadataDict

if TYPE_CHECKING:
    from strux.results import RegressionResults

@dataclass
class Experiment:
    """Represents a single experiment run."""
    id: str
    timestamp: datetime
    metadata: MetadataDict
    results: 'RegressionResults'
    
    @property
    def is_annotation_based(self) -> bool:
        """Check if the experiment is annotation-based."""
        return bool(self.results.config.annotation_field)