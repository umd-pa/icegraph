# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self, Any, TypeVar, TYPE_CHECKING, Iterator
from pathlib import Path

import torch
from torch import Tensor
import yaml
import numpy as np

# types
from icegraph.common.data import DataRole
from icegraph.common.engine import Engine
from icegraph.typing.common import ArrayF

# services
from icegraph.engine.services import ServiceManager

# components
from icegraph.engine.components import Component

from icegraph.engine.components.adapter import Adapter, AdapterFactory, AdapterContext
from icegraph.engine.components.normalizer import Normalizer, NormalizerFactory, NormalizerContext
from icegraph.engine.components.model import Model, ModelFactory, ModelContext
from icegraph.engine.components.transformer import Transformer, TransformerFactory, TransformerContext

# callbacks
from .callbacks import CallbackManager, context

# config
from .config import InferenceConfig

if TYPE_CHECKING:
    from icegraph.common.plugins import Plugin

__all__ = ["Inference"]


# module logger
import logging
logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Component[Any, Any])


class Inference(Engine):

    def __init__(
            self,
            model_path: str | Path,
            *,
            config: dict[str, Any]
    ) -> None:
        """Initialize the Trainer."""
        logger.debug("initializing %s", type(self).__name__)

        # stash outdir and derive logdir
        self.outdir = Path(model_path).parent
        self.logdir = self.outdir / "logs"

        # laod state
        self.model_state = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        # load inference config
        # allow user to overwrite model config, user must define service configs
        state_config = self.model_state["config"]
        state_config.update(config)
        self.config = InferenceConfig(**state_config)

        # extract policy
        self.policy = self.config.policy

        # initialize the service manager
        self.services   = ServiceManager.from_config(self, dict(self.config.services))

        # get defaults
        self.state      = self.services.require("state",    required_by=Inference)
        self.record     = self.services.require("record",   required_by=Inference)
        self.decode     = self.services.require("decode",   required_by=Inference)

        # build components ordered properly
        self.adapter        = self._init_adapter()
        self.transformer    = self._init_transformer()

        self.model          = self._init_model()
        self.normalizer     = self._init_normalizer()

        # initialize the callback manager
        self.callbacks = CallbackManager(self)

    @classmethod
    def from_yaml(cls, model_path: str | Path, config_path: str | Path) -> Self:
        with Path(config_path).open("r") as f:
            return cls(model_path, config=yaml.safe_load(f))

    ### POLICY HANDLER

    def ensure_compatible(self, p: Plugin, /) -> None:
        if not p.compatible:
            # an empty compatibility tuple indicates compatibility with any policy
            return

        if self.policy not in p.compatible:
            raise RuntimeError(f"Module '{type(p).__name__}' is not compatible with policy '{self.policy}'.")

    ### INIT METHODS

    def _component_payload(self, key: str) -> tuple[str, dict[str, Tensor]]:
        if key not in self.model_state:
            raise KeyError(f"Checkpoint is missing required component '{key}'.")

        payload = self.model_state[key]

        if not isinstance(payload, tuple) or len(payload) != 2:
            raise TypeError(
                f"Expected checkpoint component '{key}' to be a tuple "
                f"of (class_name, state_dict), got {type(payload)}."
            )

        return payload

    def _init_adapter(self) -> Adapter[Any]:
        """Initialize the adapter for set policy."""
        # build directly since using policy not configurations
        config = self.config.components.adapter
        adapter = AdapterFactory.create(self.policy, **config.kwargs)

        # attach the adapter
        ctx = AdapterContext(self.services)
        adapter.attach(ctx)

        # load state
        state_dict = self._component_payload("adapter")[1]
        adapter.load_state_dict(state_dict)

        # move to device
        adapter.to(self.state.device)

        return adapter

    def _init_transformer(self) -> Transformer[Any]:
        """Initialize the transformer."""
        config = self.config.components.transformer

        # create
        transformer = TransformerFactory.create(config.name, **config.kwargs)

        # attach using context
        ctx = TransformerContext(services=self.services)
        transformer.attach(ctx)

        # load state
        state_dict = self._component_payload("transformer")[1]
        transformer.load_state_dict(state_dict)

        # move to device
        transformer.to(self.state.device)

        # ensure compatible with selected policy
        self.ensure_compatible(transformer)

        return transformer

    def _init_normalizer(self) -> Normalizer[Any]:
        """Initialize the normalizer."""
        config = self.config.components.normalizer

        # create
        normalizer = NormalizerFactory.create(config.name, **config.kwargs)

        # attach using context
        ctx = NormalizerContext(
            services=self.services,
            transformer_spec_list=self.transformer.spec_list,
        )
        normalizer.attach(ctx)

        # load state
        state_dict = self._component_payload("normalizer")[1]
        normalizer.load_state_dict(state_dict)

        # move to device
        normalizer.to(self.state.device)

        # ensure compatible with selected policy
        self.ensure_compatible(normalizer)

        return normalizer

    def _init_model(self) -> Model[Any]:
        """Initialize the model for training."""
        config = self.config.components.model

        # create
        model = ModelFactory.create(config.name, **config.kwargs)

        # attach using context
        ctx = ModelContext(
            services=self.services,
            in_channels=self.adapter.in_channels,
            out_channels=self.adapter.out_channels,
        )
        model.attach(ctx)

        # load state
        state_dict = self._component_payload("model")[1]
        model.load_state_dict(state_dict)

        # move to device
        model.to(self.state.device)

        # ensure compatible with selected policy
        self.ensure_compatible(model)

        # model needs to be bound to execution context
        self.state.bind_model(model)

        return model

    ### ---

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _fire(self, hook_name: str, ctx: context.Context) -> None:
        """
        Invokes internal callback hooks.

        Args:
            hook_name (str): The name of the callback method to call.
            ctx: Context to forward into the callback.
        """
        self.callbacks.fire(hook_name, ctx)

    @torch.no_grad()
    def predict(self, features: Tensor, batch: Tensor) -> Tensor:
        features = self.transformer(features, DataRole.FEATURES)
        features = self.normalizer(features, DataRole.FEATURES)

        out = self.model(features, batch)
        out = self.adapter.adapt_model_out(out)

        if self.adapter.use_normalized_targets:
            out = self.normalizer(out, DataRole.TARGETS, inverse=True)
            out = self.transformer(out, DataRole.TARGETS, inverse=True)

        return out.softmax(dim=-1).squeeze(1)

    @staticmethod
    def normalize_weights(weights: ArrayF, file_count: int) -> ArrayF:
        return weights / file_count

    def execute(self) -> Iterator[ArrayF]:
        """Execute inference."""
        logger.info("executing inference")

        # fire on_execute callback hook
        ctx = context.ExecuteContext(self)
        self._fire("on_execute", ctx)

        features_list = []
        weight_list = []
        batch_size = 1024

        for record in self.record:
            features = np.asarray(record.data.get("features"))
            weight_list.append(float(record.data.get("simweights")))
            features = torch.tensor(features, dtype=torch.float32)
            features_list.append(features)

            if len(features_list) == batch_size:
                features = torch.cat(features_list, dim=0).to(self.state.device)

                batch = torch.cat([
                    torch.full(
                        (features_i.shape[0],),
                        i,
                        dtype=torch.long,
                        device=self.state.device,
                    )
                    for i, features_i in enumerate(features_list)
                ])

                yield self.predict(features, batch).cpu().numpy(), self.normalize_weights(np.asarray(weight_list), self.record.file_count)

                features_list = []
                weight_list = []

        # run final partial batch
        if features_list:
            features = torch.cat(features_list, dim=0).to(self.state.device)

            batch = torch.cat([
                torch.full(
                    (features_i.shape[0],),
                    i,
                    dtype=torch.long,
                    device=self.state.device,
                )
                for i, features_i in enumerate(features_list)
            ])

            yield self.predict(features, batch).cpu().numpy(), self.normalize_weights(np.asarray(weight_list), self.record.file_count)

    def close(self) -> None:
        try:
            ctx = context.TeardownContext(self)
            self._fire("on_teardown", ctx)
        finally:
            self.services.close()
