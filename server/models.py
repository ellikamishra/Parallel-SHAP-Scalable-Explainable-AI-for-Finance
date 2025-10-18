from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column
from sqlmodel import SQLModel, Field, JSON

class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    label: str = Field(index=True)
    dataset_name: str = "market_features.csv"
    backend: str
    params: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    hardware: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    runtime_sec: float
    speedup_vs_baseline: Optional[float] = None
    fidelity_corr: Optional[float] = None
    notes: Optional[str] = None
