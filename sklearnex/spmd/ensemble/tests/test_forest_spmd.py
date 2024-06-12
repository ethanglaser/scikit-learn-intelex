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
from sklearn.datasets import make_regression

from onedal.tests.utils._spmd_support import (
    generate_classification_data,
    generate_regression_data,
    get_local_tensor,
    mpi_libs_and_gpu_available,
    spmd_assert_all_close,
)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_rfcls_spmd_manual():
    # Import spmd and batch algo
    from sklearnex.ensemble import RandomForestClassifier as RandomForestClassifier_Batch
    from sklearnex.spmd.ensemble import (
        RandomForestClassifier as RandomForestClassifier_SPMD,
    )

    # Create gold data and process into dpt
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    y_train = np.array([0, 2, 1, 2, 1, 0, 1, 2, 0])
    X_test = np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ]
    )

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure predictions of batch algo match spmd
    spmd_model = RandomForestClassifier_SPMD(n_estimators=3, random_state=0).fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = RandomForestClassifier_Batch(n_estimators=3, random_state=0).fit(
        X_train, y_train
    )
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    pytest.skip("SPMD and batch random forest results not aligned")
    spmd_assert_all_close(spmd_result, batch_result)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 1000])
@pytest.mark.parametrize("n_features_and_classes", [(5, 2), (25, 2), (25, 10)])
@pytest.mark.parametrize("n_estimators", [10, 100])
@pytest.mark.parametrize("max_depth", [3, None])
@pytest.mark.mpi
def test_rfcls_spmd_synthetic(n_samples, n_features_and_classes, n_estimators, max_depth):
    n_features, n_classes = n_features_and_classes
    # Import spmd and batch algo
    from sklearnex.ensemble import RandomForestClassifier as RandomForestClassifier_Batch
    from sklearnex.spmd.ensemble import (
        RandomForestClassifier as RandomForestClassifier_SPMD,
    )

    # Generate data and process into dpt
    X_train, X_test, y_train, _ = generate_classification_data(
        n_samples, n_features, n_classes
    )

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure predictions of batch algo match spmd
    spmd_model = RandomForestClassifier_SPMD(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    ).fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = RandomForestClassifier_Batch(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    ).fit(X_train, y_train)
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    pytest.skip("SPMD and batch random forest results not aligned")
    spmd_assert_all_close(spmd_result, batch_result)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.mpi
def test_rfreg_spmd_manual():
    # Import spmd and batch algo
    from sklearnex.ensemble import RandomForestRegressor as RandomForestRegressor_Batch
    from sklearnex.spmd.ensemble import (
        RandomForestRegressor as RandomForestRegressor_SPMD,
    )

    # Create gold data and process into dpt
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    y_train = np.array([3.0, 5.0, 4.0, 7.0, 5.0, 6.0, 1.0, 2.0, 0.0])
    X_test = np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ]
    )

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure predictions of batch algo match spmd
    spmd_model = RandomForestRegressor_SPMD(n_estimators=3, random_state=0).fit(
        local_dpt_X_train, local_dpt_y_train
    )
    batch_model = RandomForestRegressor_Batch(n_estimators=3, random_state=0).fit(
        X_train, y_train
    )
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    pytest.skip("SPMD and batch random forest results not aligned")
    spmd_assert_all_close(spmd_result, batch_result)


@pytest.mark.skipif(
    not mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize("n_samples", [200, 1000])
@pytest.mark.parametrize("n_features", [5, 25])
@pytest.mark.parametrize("n_estimators", [10, 100])
@pytest.mark.parametrize("max_depth", [3, None])
@pytest.mark.mpi
def test_rfreg_spmd_synthetic(n_samples, n_features, n_estimators, max_depth):
    # Import spmd and batch algo
    from sklearnex.ensemble import RandomForestRegressor as RandomForestRegressor_Batch
    from sklearnex.spmd.ensemble import (
        RandomForestRegressor as RandomForestRegressor_SPMD,
    )

    # Generate data and process into dpt
    X_train, X_test, y_train, _ = generate_regression_data(n_samples, n_features)

    local_dpt_X_train = get_local_tensor(X_train)
    local_dpt_y_train = get_local_tensor(y_train)
    local_dpt_X_test = get_local_tensor(X_test)

    # ensure predictions of batch algo match spmd
    spmd_model = RandomForestRegressor_Batch(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    ).fit(local_dpt_X_train, local_dpt_y_train)
    batch_model = RandomForestRegressor_Batch(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    ).fit(X_train, y_train)
    spmd_result = spmd_model.predict(local_dpt_X_test)
    batch_result = batch_model.predict(X_test)

    # TODO: remove skips when SPMD and batch are aligned
    pytest.skip("SPMD and batch random forest results not aligned")
    spmd_assert_all_close(spmd_result, batch_result)