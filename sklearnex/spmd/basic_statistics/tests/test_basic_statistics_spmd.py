# ==============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onedal.tests.utils._spmd_support import mpi_libs_and_gpu_available, get_local_tensor, generate_statistic_data


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_basic_stats_spmd_manual():
    # Import spmd and batch algo
    from onedal.basic_statistics import BasicStatistics as BasicStatistics_Batch
    from sklearnex.spmd.basic_statistics import BasicStatistics as BasicStatistics_SPMD

    # Create gold data and process into dpt
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0],
            [0.0, 3.0, 8.0],
            [0.0, 4.0, 16.0],
            [0.0, 5.0, 32.0],
            [0.0, 6.0, 64.0],
        ]
    )

    local_dpt_data = get_local_tensor(data)

    # ensure results of batch algo match spmd
    spmd_result = BasicStatistics_SPMD().compute(local_dpt_data)
    batch_result = BasicStatistics_Batch().compute(data)

    for option in batch_result.keys():
        assert_allclose(spmd_result[option], batch_result[option])


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [100, 10000])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.mpi
def test_basic_stats_spmd_synthetic(n_samples, n_features):
    # Import spmd and batch algo
    from onedal.basic_statistics import BasicStatistics as BasicStatistics_Batch
    from sklearnex.spmd.basic_statistics import BasicStatistics as BasicStatistics_SPMD

    # Generate data and process into dpt
    data = generate_statistic_data(n_samples, n_features)

    local_dpt_data = get_local_tensor(data)

    # ensure results of batch algo match spmd
    spmd_result = BasicStatistics_SPMD().compute(local_dpt_data)
    batch_result = BasicStatistics_Batch().compute(data)

    for option in batch_result.keys():
        assert_allclose(spmd_result[option], batch_result[option])
