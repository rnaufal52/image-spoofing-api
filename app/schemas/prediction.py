
from pydantic import BaseModel
from typing import List, Optional

class PredictionResult(BaseModel):
    decision: str
    mean_score: float
    patch_max_score: float
    local_variance: float
    rgb_corr: float
    freq_spike_ratio: float
    glare_asym: float
    evidence_count: int
    evidence: List[str]
    reason: str

class AntiSpoofResponse(BaseModel):
    success: bool
    result: PredictionResult
