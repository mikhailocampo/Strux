"""Common types used across the package."""
from typing import TypeVar, Dict, Any
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
MetadataDict = Dict[str, Any] 