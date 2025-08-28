# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import List, Optional, Dict, Self, Literal
from pydantic import BaseModel, Field, validator, model_validator


# === io ===
class IOConfig(BaseModel):
    default_dir: str
    gcd_path: str


# === frame_keys ===
class FrameKeysConfig(BaseModel):
    header: str
    mctree: str
    bg_mctree: str
    weight_dict: str
    truth_dict: str


# === table_names ===
class TableNamesConfig(BaseModel):
    features: str
    truth: str


# === data ===
class DataSplitsConfig(BaseModel):
    weights: Dict[Literal['train', 'val', 'test'], int]

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if not self.weights:
            raise ValueError("Weights must not be empty.")
        if any(v < 0 for v in self.weights.values()):
            raise ValueError("Weights values must be non-negative.")
        if sum(self.weights.values()) <= 0:
            raise ValueError("Sum of weights must be > 0.")
        return self


class DataNormalizationConfig(BaseModel):
    apply_log_scaling_x: List[str] = Field(default_factory=list)
    apply_log_scaling_y: List[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    target_labels: List[str]
    splits: DataSplitsConfig
    normalization: DataNormalizationConfig


# === training ===
class TrainerParamsConfig(BaseModel):
    max_epochs: int
    num_nbrs: int
    hidden_layers: int
    hidden_channels: int
    val_interval_epochs: int


class OptimizerConfig(BaseModel):
    learning_rate: float
    betas: List[float]
    eps: float
    weight_decay: float
    amsgrad: bool

    @model_validator(mode="after")
    def validate_betas(self) -> Self:
        if len(self.betas) != 2:
            raise ValueError("Optimizer 'betas' must be a list of two floats.")
        return self


class TensorBoardConfig(BaseModel):
    port: int


class TrainingConfig(BaseModel):
    seed: int
    batch_size: int
    num_workers: int
    trainer_params: TrainerParamsConfig
    normalizer: str
    optimizer: OptimizerConfig
    tensorboard: TensorBoardConfig


# === feature_extraction ===
class DOMExclusionsConfig(BaseModel):
    exclusions: List[str] = Field(default_factory=list)
    partial_exclusion: bool


class PulseModifierConfig(BaseModel):
    class_: Optional[str] = Field(default=None, alias="class")
    kwargs: Dict = Field(default_factory=dict)


class FeatureEntry(BaseModel):
    class_: str = Field(..., alias="class")
    kwargs: Optional[Dict] = Field(default_factory=dict)


class FeatureConfig(BaseModel):
    features: List[FeatureEntry]


class FeatureExtractionConfig(BaseModel):
    pulse_key: str
    dom_exclusions: DOMExclusionsConfig
    pulse_modifier: PulseModifierConfig
    feature_config: FeatureConfig


# === Root config ===
class FullConfig(BaseModel):
    io: IOConfig
    frame_keys: FrameKeysConfig
    table_names: TableNamesConfig
    data: DataConfig
    training: TrainingConfig
    feature_extraction: FeatureExtractionConfig
