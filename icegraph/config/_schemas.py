# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import List, Optional, Dict, Self, Literal, Any, ClassVar
from pydantic import BaseModel, Field, model_validator


# === io ===
class IOConfig(BaseModel):
    default_dir: str
    gcd_path: str


# === frame_keys ===
class FrameKeys(BaseModel):
    header: str
    mctree: str
    truth_dict: str
    bg_mctree: Optional[str] = None
    weight_dict: Optional[str] = None
    corsika_weight_map: Optional[str] = None


# === frames ===
class FrameConfig(BaseModel):
    corsika: bool
    frame_keys: FrameKeys

    @model_validator(mode="after")
    def check_dependencies(self):
        if self.corsika:
            if not self.frame_keys.corsika_weight_map:
                raise ValueError("When corsika = True: corsika_weight_map is required.")
        else:
            if not self.frame_keys.bg_mctree or not self.frame_keys.weight_dict:
                raise ValueError("When corsika = False: bg_mctree, weight_dict, and truth_dict are required.")
        return self


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
    labels: List[str]
    splits: DataSplitsConfig


# === training ===
class TrainerConfig(BaseModel):
    max_epochs: int
    val_interval_epochs: int
    save_interval: int

class LoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    prefetch_factor: int
    mp_context: Literal["fork", "forkserver", "spawn"]

class ModelConfig(BaseModel):
    task: str
    kwargs: Dict[str, Any]


class OptimizerConfig(BaseModel):
    task: str
    kwargs: Dict[str, Any]


class StrategyConfig(BaseModel):
    task: str
    kwargs: Dict[str, Any]


class SchedulerConfig(BaseModel):
    task: Optional[str]
    step_mode: Literal['batch', 'epoch', 'warm_restarts']
    kwargs: Dict[str, Any]


class TensorBoardConfig(BaseModel):
    port: int


class NormalizerConfig(BaseModel):
    task: str
    kwargs: DataNormalizationConfig


class MetricConfig(BaseModel):
    task: str
    kwargs: Dict[str, Any]


class TrainingConfig(BaseModel):
    # global seed for reproducibility
    seed:               int

    # labels
    target_labels:      List[str]
    auxiliary_labels:   List[str]

    # support multiple options
    metrics:            List[MetricConfig]

    # support only one option
    trainer:            TrainerConfig
    loader:             LoaderConfig
    model:              ModelConfig
    strategy:           StrategyConfig
    normalizer:         NormalizerConfig
    optimizer:          OptimizerConfig
    scheduler:          SchedulerConfig
    tensorboard:        TensorBoardConfig


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
    frames: FrameConfig
    table_names: TableNamesConfig
    data: DataConfig
    training: TrainingConfig
    feature_extraction: FeatureExtractionConfig
