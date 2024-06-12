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

from onedal.tests.utils._spmd_support import (
    generate_statistic_data,
    get_local_tensor,
    mpi_libs_and_gpu_available,
)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_covariance_spmd_manual():
    # Import spmd and batch algo
    from onedal.covariance import EmpiricalCovariance as EmpiricalCovariance_Batch
    from sklearnex.spmd.covariance import EmpiricalCovariance as EmpiricalCovariance_SPMD

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
    spmd_result = EmpiricalCovariance_SPMD().fit(local_dpt_data)
    batch_result = EmpiricalCovariance_Batch().fit(data)

    assert_allclose(spmd_result.covariance_, batch_result.covariance_)
    assert_allclose(spmd_result.location_, batch_result.location_)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [100, 10000])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.parametrize("assume_centered", [True, False])
@pytest.mark.mpi
def test_covariance_spmd_synthetic(n_samples, n_features, assume_centered):
    # Import spmd and batch algo
    from onedal.covariance import EmpiricalCovariance as EmpiricalCovariance_Batch
    from sklearnex.spmd.covariance import EmpiricalCovariance as EmpiricalCovariance_SPMD

    # Generate data and process into dpt
    data = generate_statistic_data(n_samples, n_features)

    local_dpt_data = get_local_tensor(data)

    # ensure results of batch algo match spmd
    spmd_result = EmpiricalCovariance_SPMD(assume_centered=assume_centered).fit(
        local_dpt_data
    )
    batch_result = EmpiricalCovariance_Batch(assume_centered=assume_centered).fit(data)

    assert_allclose(spmd_result.covariance_, batch_result.covariance_)
    assert_allclose(spmd_result.location_, batch_result.location_)