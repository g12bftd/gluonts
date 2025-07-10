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

"""Alternating Transformer model components."""


# Relative imports
from ._estimator import (
    projection_mat,
    AlternatingHierarchicalTransformerEstimator,
)
from gluonts.mx.model.deepvar_hierarchical._network import (
    reconcile_samples,
    coherency_error
)
from  gluonts.mx.model.deepvar_hierarchical._estimator import projection_mat

__all__ = [
    "AlternatingHierarchicalTransformerEstimator",
    "projection_mat",
    "reconcile_samples",
    "coherency_error",
]
