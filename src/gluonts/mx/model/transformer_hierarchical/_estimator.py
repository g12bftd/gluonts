from functools import partial
from typing import List, Optional
import logging

from mxnet.gluon import HybridBlock
import mxnet as mx
import numpy as np

from gluonts.transform._base import MapTransformation
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.distribution import DistributionOutput, StudentTOutput, LowrankMultivariateGaussianOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)

from ._network import HierarchicalTransformerPredictionNetwork, HierarchicalTransformerTrainingNetwork
from .transformer_encoder import HierarchicalTransformerEncoder
from .transformer_decoder import HierarchicalTransformerDecoder

from gluonts.mx.model.deepvar_hierarchical._estimator import projection_mat

logger = logging.getLogger(__name__)

class HierarchicalTransformerEstimator(GluonEstimator):

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        S: np.ndarray,
        D: Optional[np.ndarray] = None,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_samples_for_loss: int = 200,
        likelihood_weight: float = 0.0,
        CRPS_weight: float = 1.0,
        sample_LH: bool = False,
        coherent_train_samples: bool = True,
        coherent_pred_samples: bool = True,
        warmstart_epoch_frac: float = 0.0,
        seq_axis: Optional[List[int]] = None,
        log_coherency_error: bool = True,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        dropout_rate: float = 0.1,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: int = 20,
        distr_output: DistributionOutput = StudentTOutput(),
        model_dim: int = 32,
        inner_ff_dim_scale: int = 4,
        pre_seq: str = "dn",
        post_seq: str = "drn",
        act_type: str = "softrelu",
        num_heads: int = 8,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        num_parallel_samples: int = 100,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        batch_size: int = 32,
    ) -> None:
        rank = 0
        target_dim = len(S)
        self.target_dim = target_dim
        distr_output = LowrankMultivariateGaussianOutput(
            dim=target_dim, rank=rank
        )
        super().__init__(trainer=trainer, batch_size=batch_size)

        # Assert that projection is *not* being done only during training
        assert coherent_pred_samples or (
            not coherent_train_samples
        ), "Cannot project only during training (and not during prediction)"


        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"
        assert (
            cardinality is not None or not use_feat_static_cat
        ), "You must set `cardinality` if `use_feat_static_cat=True`"
        assert cardinality is None or all(
            [c > 0 for c in cardinality]
        ), "Elements of `cardinality` should be > 0"
        assert (
            embedding_dimension > 0
        ), "The value of `embedding_dimension` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"

        M = projection_mat(S=S, D=D)
        self.S = S
        ctx = self.trainer.ctx
        self.M = mx.nd.array(M, ctx=ctx)
        self.num_samples_for_loss = num_samples_for_loss
        self.likelihood_weight = likelihood_weight
        self.CRPS_weight = CRPS_weight
        self.log_coherency_error = log_coherency_error
        self.coherent_train_samples = coherent_train_samples
        self.coherent_pred_samples = coherent_pred_samples
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.sample_LH = sample_LH
        self.seq_axis = seq_axis


        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.distr_output = distr_output
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.cardinality = cardinality if use_feat_static_cat else [1]
        self.embedding_dimension = embedding_dimension
        self.num_parallel_samples = num_parallel_samples
        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_lags_for_frequency(freq_str=freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(freq)
        )
        self.history_length = self.context_length + max(self.lags_seq)
        self.scaling = scaling

        self.encoder_config = {
            "model_dim": model_dim,
            "pre_seq": pre_seq,
            "post_seq": post_seq,
            "dropout_rate": dropout_rate,
            "inner_ff_dim_scale": inner_ff_dim_scale,
            "act_type": act_type,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers,
            "num_series": self.S.shape[0],
            "encoder_length": context_length,
            "attention_scheme": "full",
        }

        self.decoder_config = {
            "model_dim": model_dim,
            "pre_seq": pre_seq,
            "post_seq": post_seq,
            "dropout_rate": dropout_rate,
            "inner_ff_dim_scale": inner_ff_dim_scale,
            "act_type": act_type,
            "num_heads": num_heads,
            "num_decoder_layers": num_decoder_layers,
            "num_series": self.S.shape[0],
            "decoder_length": prediction_length,
            "attention_scheme": "full",
        }

        self.encoder = HierarchicalTransformerEncoder(
            self.encoder_config, prefix="enc_"
        )
        self.decoder = HierarchicalTransformerDecoder(
            self.decoder_config, prefix="dec_"
        )
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

    def _token_builder(self) -> Transformation:
        """
        Wrap self._build_enc_dec_tokens(data) so it can sit in a Chain.
        """
    
        # --- define an inner class *inside the method* -----------------
        estimator = self # capture outer `self` for closure
    
        class _BuildTokens(MapTransformation):
            def map_transform(inner_self, data, is_train: bool):
                enc, dec = estimator.build_enc_dec_tokens(data)
                data["enc_tokens"] = enc
                data["dec_tokens"] = dec
                return data

        return _BuildTokens()


    def build_enc_dec_tokens(self, data):
        """
        Build the tensors the encoder and decoder will consume.
    
        Parameters
        ----------
        data : dict
            One training instance produced by InstanceSplitter.
            Must contain at least the fields printed in your BATCH SHAPES:
                past_target, past_time_feat, past_observed_values,
                future_time_feat, future_observed_values,
                feat_static_cat.
    
        Returns
        -------
        enc_tokens : mx.nd.NDArray, shape (B, N, T, 4)
        dec_tokens : mx.nd.NDArray, shape (B, N, L, 4)
            where
                B = batch size
                N = 89 nodes in the hierarchy  (rows of S)
                T = context_length             (37 quarters here)
                L = prediction_length          (4 quarters here)
            The last dimension packs 4 per‑step features:
                0. target (or lag‑1 value for decoder)
                1. calendar covariate          (quarter id / age)
                2. observed‑values mask        (1 = present, 0 = missing)
                3. static category id          (broadcast)
        """
    
        # ---------- tensors directly from the batch -----------------------
        past_target   = data["past_target"]          # (B, T, N)
        past_obs      = data["past_observed_values"]
        past_timefeat = data["past_time_feat"]       # (B, T, 1)
    
        future_obs      = data["future_observed_values"]
        future_timefeat = data["future_time_feat"]
        static_cat = data[FieldName.FEAT_STATIC_CAT] # th
    
        # ---------- dimensions & context ---------------------------------

        print(f"past_target shape: {past_target.shape}")
        print(f"future_obs shape: {future_obs.shape}")
        B, T, N = past_target.shape
        
        L       = future_obs.shape[1]            # prediction_length
       
        ctx     = past_target.context            # gpu(0) or cpu(0)
    
        # ---------- static category  (B,1) -> (B,N,1) --------------------
        static_cat = data["feat_static_cat"] .expand_dims(1)   # (B,1,1)
        static_cat = mx.nd.broadcast_to(static_cat, shape=(B, N, 1))  # (B,N,1)
    
        stc_enc = mx.nd.broadcast_to(static_cat.expand_dims(-1), (B, N, T, 1))
        stc_dec = mx.nd.broadcast_to(static_cat.expand_dims(-1), (B, N, L, 1))
    
        # ---------- calendar features ------------------------------------
        # past: (B,T,1) -> (B,1,T,1) -> (B,N,T,1)
        ptf = past_timefeat.transpose((0, 2, 1)).expand_dims(-1)       # (B,1,T,1)
        ptf = mx.nd.broadcast_to(ptf, shape=(B, N, T, 1))
    
        # future: (B,L,1) -> (B,1,L,1) -> (B,N,L,1)
        ftf = future_timefeat.transpose((0, 2, 1)).expand_dims(-1)     # (B,1,L,1)
        ftf = mx.nd.broadcast_to(ftf, shape=(B, N, L, 1))
    
        # ---------- observed‑value masks ---------------------------------
        pob = past_obs.transpose((0, 2, 1)).expand_dims(-1)            # (B,N,T,1)
        fob = future_obs.transpose((0, 2, 1)).expand_dims(-1)          # (B,N,L,1)
    
        # ---------- target channels --------------------------------------
        # encoder uses the true past
        ptt = past_target.transpose((0, 2, 1)).expand_dims(-1)         # (B,N,T,1)
    
        # decoder starts with the last context value (lag‑1), repeated over L
        last_ctx = past_target[:, -1, :].expand_dims(-1).expand_dims(-1)  # (B,N,1,1)
        last_ctx = mx.nd.broadcast_to(last_ctx, shape=(B, N, L, 1))       # (B,N,L,1)
    
        # ---------- concatenate features ---------------------------------
        enc_tokens = mx.nd.concat(ptt, ptf, pob, stc_enc, dim=-1)      # (B,N,T,4)
        dec_tokens = mx.nd.concat(last_ctx, ftf, fob, stc_dec, dim=-1) # (B,N,L,4)
    
        return enc_tokens.astype("float32"), dec_tokens.astype("float32")



    def _basic_feature_chain(self) -> Transformation:
        remove_field_names = [
            FieldName.FEAT_DYNAMIC_CAT,
            FieldName.FEAT_STATIC_REAL,
        ]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat
                else []
            )
            + [
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
            ]
        )

    def _create_instance_splitter(self, mode: str):
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
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        )

    def _full_transform(self, mode: str) -> Transformation:
        return (
            self._basic_feature_chain()          # feature engineering
            + self._create_instance_splitter(mode)  # windowing (training / val / test)
            + self._token_builder()                 # adds "enc_tokens" & "dec_tokens"
        )


    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
    
        input_names = get_hybrid_forward_input_names(
            HierarchicalTransformerTrainingNetwork
        )
        wanted = ["enc_tokens", "dec_tokens", "future_target"]
        transform = self._full_transform("training") + SelectFields(wanted)
        return TrainDataLoader(
            dataset=data,
            transform=transform,
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            HierarchicalTransformerTrainingNetwork
        )
        #instance_splitter = self._create_instance_splitter("validation")
        wanted = ["enc_tokens", "dec_tokens", "future_target"]
        transform = self._full_transform("validation") + SelectFields(wanted)
        return ValidationDataLoader(
            dataset=data,
            transform=transform,
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> HierarchicalTransformerTrainingNetwork:
        return HierarchicalTransformerTrainingNetwork(
            M=self.M,
            S=self.S,
            num_samples_for_loss=self.num_samples_for_loss,
            likelihood_weight=self.likelihood_weight,
            CRPS_weight=self.CRPS_weight,
            seq_axis=self.seq_axis,
            coherent_train_samples=self.coherent_train_samples,
            warmstart_epoch_frac=self.warmstart_epoch_frac,
            epochs=self.trainer.epochs,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            sample_LH=self.sample_LH,
            target_dim=self.target_dim,
            encoder=self.encoder,
            decoder=self.decoder,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = HierarchicalTransformerPredictionNetwork(
            M=self.M,
            S=self.S,
            log_coherency_error=self.log_coherency_error,
            coherent_pred_samples=self.coherent_pred_samples,
            target_dim=self.target_dim,
            encoder=self.encoder,
            decoder=self.decoder,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )

