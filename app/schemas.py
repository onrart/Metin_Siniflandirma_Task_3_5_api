from typing import List
from pydantic import BaseModel, Field


class PredictParams(BaseModel):
    multi_label: bool = False
    threshold: float = 0.5
    top_k: int = 3
    max_length: int = 512
    truncation: bool = True
    pipeline_batch_size: int = 32


class SinglePredictRequest(BaseModel):
    text: str
    params: PredictParams = Field(default_factory=PredictParams)


class BatchPredictRequest(BaseModel):
    texts: List[str]
    params: PredictParams = Field(default_factory=PredictParams)


class PredictionItem(BaseModel):
    label: str
    score: float


class SinglePredictResponse(BaseModel):
    text: str
    predictions: List[PredictionItem]


class BatchPredictResponse(BaseModel):
    results: List[SinglePredictResponse]
