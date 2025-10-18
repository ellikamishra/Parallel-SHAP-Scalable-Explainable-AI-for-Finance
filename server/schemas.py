from pydantic import BaseModel
from typing import Optional

class BenchmarkRequest(BaseModel):
    label: str
    backend: str
    P: int = 64
    threads: Optional[int] = None
    N: Optional[int] = None
    dataset_name: str = "market_features.csv"
    notes: Optional[str] = None
