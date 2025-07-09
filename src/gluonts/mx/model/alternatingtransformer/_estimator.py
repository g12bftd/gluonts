# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Estimator definition for the Alternating Transformer model."""

from typing import Optional, List
from functools import partial

from mxnet.gluon import HybridBlock
from gluonts.model.predictor import Predictor

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.mx.batchify import batchify
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import get_hybrid_forward_input_names
from gluonts.transform import (
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)
from gluonts.transform.feature import DummyValueImputation

from ._network import (
    AlternatingTransformerNetwork,
    AlternatingTransformerTrainingNetwork,
    AlternatingTransformerPredictionNetwork,
)


class AlternatingTransformerEstimator(GluonEstimator):
    """A minimal estimator for the Alternating Transformer model."""

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        num_series: int,
        num_layers: int = 2,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        distr_output: DistributionOutput = StudentTOutput(),
        num_parallel_samples: int = 100,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        batch_size: int = 32,
        debug: bool = False,
    ) -> None:
        super().__init__(trainer=trainer, batch_size=batch_size)

        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.num_series = num_series
        self.num_layers = num_layers
        self.distr_output = distr_output
        self.num_parallel_samples = num_parallel_samples
        self.debug = debug
        self.train_sampler = (
            train_sampler
            if train_sampler is not None
            else ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=prediction_length
            )
        )
        self.validation_sampler = (
            validation_sampler
            if validation_sampler is not None
            else ValidationSplitSampler(min_future=prediction_length)
        )

        self.config = {
            "model_dim": 32,
            "pre_seq": "dn",
            "post_seq": "drn",
            "dropout_rate": 0.1,
            "inner_ff_dim_scale": 4,
            "act_type": "softrelu",
            "num_heads": 4,
        }

    def create_transformation(self) -> Transformation:
        return SelectFields(
            [
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
            ],
            allow_missing=True,
        ) + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
            dtype=self.dtype,
            imputation_method=DummyValueImputation(
                self.distr_output.value_in_support
            ),
        )

    def _create_instance_splitter(self, mode: str) -> InstanceSplitter:
        assert mode in ["training", "validation", "test"]
        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )

    def create_training_data_loader(
        self, data: Dataset, **kwargs
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            AlternatingTransformerTrainingNetwork
        )
        instance_splitter = self._create_instance_splitter("training")
        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            **kwargs,
        )

    def create_validation_data_loader(
        self, data: Dataset, **kwargs
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            AlternatingTransformerTrainingNetwork
        )
        instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> HybridBlock:
        base_net = AlternatingTransformerNetwork(
            num_layers=self.num_layers,
            num_series=self.num_series,
            num_timesteps=self.context_length,
            config=self.config,
            debug=self.debug,
        )
        return AlternatingTransformerTrainingNetwork(
            base_network=base_net,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        base_net = AlternatingTransformerNetwork(
            num_layers=self.num_layers,
            num_series=self.num_series,
            num_timesteps=self.context_length,
            config=self.config,
            debug=self.debug,
        )
        prediction_network = AlternatingTransformerPredictionNetwork(
            base_network=base_net,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            num_samples=self.num_parallel_samples,
            params=trained_network.collect_params(),
        )
        prediction_splitter = self._create_instance_splitter("test")
        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            forecast_generator=DistributionForecastGenerator(
                self.distr_output
            ),
            ctx=self.trainer.ctx,
        )
