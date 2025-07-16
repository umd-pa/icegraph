# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, validator


# === io ===
class IOConfig(BaseModel):
    default_dir: str
    gcd_path: str


# === frame_keys ===
class FrameKeysConfig(BaseModel):
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
    stratify: bool


class DataNormalizationConfig(BaseModel):
    apply_log_scaling: List[str] = Field(default_factory=list)


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
    test_interval_epochs: int


class OptimizerConfig(BaseModel):
    learning_rate: float
    betas: List[float]
    eps: float
    weight_decay: float
    amsgrad: bool

    @validator("betas")
    def betas_must_have_two_elements(cls, v):
        if len(v) != 2:
            raise ValueError("Optimizer 'betas' must be a list of two floats.")
        return v


class TrainingConfig(BaseModel):
    seed: int
    batch_size: int
    num_workers: int
    trainer_params: TrainerParamsConfig
    optimizer: OptimizerConfig


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
