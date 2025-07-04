<!--
******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

  
# Installation  <!-- omit in toc -->

To install Extension for Scikit-learn*, use one of the following scenarios:

- [Before You Begin](#before-you-begin)
- [Install via PIP](#install-via-pip)
  - [Install from PyPI Channel (recommended by default)](#install-from-pypi-channel-recommended-by-default)
- [Install via conda](#install-via-conda)
  - [Install from Conda-Forge Channel](#install-from-conda-forge-channel)
  - [Install from Intel conda Channel](#install-from-intel-conda-channel)
- [Build from Sources](#build-from-sources)
  - [Prerequisites](#prerequisites)
  - [Configure the Build with Environment Variables](#configure-the-build-with-environment-variables)
  - [Build Extension for Scikit-learn](#build-intelr-extension-for-scikit-learn)
- [Build from Sources with `conda-build`](#build-from-sources-with-conda-build)
  - [Prerequisites for `conda-build`](#prerequisites-for-conda-build)
  - [Build Extension for Scikit-learn with `conda-build`](#build-intelr-extension-for-scikit-learn-with-conda-build)
- [Next Steps](#next-steps)


## Before You Begin

Check [System](https://uxlfoundation.github.io/scikit-learn-intelex/latest/system-requirements.html) and [Memory](https://uxlfoundation.github.io/scikit-learn-intelex/latest/memory-requirements.html) Requirements.

## Supported Configurations

* Operating systems: Linux*, Windows*
* Python versions: 3.9 through 3.13
* Devices: CPU, GPU
* Distribution channels:
  * PyPI
  * Conda-Forge Channel
  * Intel conda Channel (https://software.repos.intel.com/python/conda/)

## Install via PIP

To prevent version conflicts, create and activate a new environment:

   - On Linux:

     ```bash
     python -m venv env
     source env/bin/activate
     ```

   - On Windows:

     ```bash
     python -m venv env
     .\env\Scripts\activate
     ```

### Install from PyPI Channel (recommended by default)

Install `scikit-learn-intelex`:

   ```bash
   pip install scikit-learn-intelex
   ```

## Install via conda

To prevent version conflicts, we recommend to create and activate a new environment. 

### Install from Conda-Forge Channel

- Install into a newly created environment (recommended):

  ```bash
  conda create -n sklex -c conda-forge --override-channels scikit-learn-intelex
  conda activate sklex
  ```

- Install into your current environment:

  ```bash
  conda install -c conda-forge scikit-learn-intelex
  ```

### Install Intel conda Channel

We recommend this installation for the users of Intel® Distribution for Python.

- Install into a newly created environment (recommended):

  ```bash
  conda create -n sklex -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels scikit-learn-intelex
  conda activate sklex
  ```

- Install into your current environment:

  ```bash
  conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge scikit-learn-intelex
  ```

**Note:** packages from the Intel channel are meant to be used together with dependencies from the **conda-forge** channel, and might not
work correctly when used in an environment where packages from the `anaconda` default channel have been installed. It is
advisable to use the [miniforge](https://github.com/conda-forge/miniforge) installer for `conda`/`mamba`, as it comes with
`conda-forge` as the only default channel.


## Build from Sources
Extension for Scikit-learn* is easily built from the sources with the majority of the necessary prerequisites available with conda or pip. 

The package is available for Windows* OS, Linux* OS, and macOS*.

Multi-node (distributed) and streaming support can be disabled if needed.

The build-process (using setup.py) happens in 4 stages:
1. Creating C++ and Cython sources from oneDAL C++ headers
2. Building oneDAL Python interfaces via cmake and pybind11
3. Running Cython on generated sources
4. Compiling and linking them

### Prerequisites
* Python version >= 3.9
* Jinja2
* Cython
* Numpy
* cmake and pybind11
* A C++ compiler with C++11 support
* Clang-Format version >=14
* [oneAPI Data Analytics Library (oneDAL)](https://github.com/uxlfoundation/oneDAL) version 2021.1 or later, but be mindful that **the oneDAL version must be <= than that of scikit-learn-intelex** (it's backwards compatible but not forwards compatible).
  * You can use the pre-built `dal-devel` conda package from conda-forge channel
* MPI (optional, needed for distributed mode)
  * You can use the pre-built `impi_rt` and `impi-devel` conda packages from conda-forge channel
* A DPC++ compiler (optional, needed for DPC++ interfaces)
  * Note that this also requires a oneDAL build with DPC++ enabled.

### Configure the Build with Environment Variables
* ``SKLEARNEX_VERSION``: sets the package version
* ``DALROOT``: sets the oneAPI Data Analytics Library path
* ``MPIROOT``: sets the path to the MPI library that will be used for distributed mode support. If this variable is not set but `I_MPI_ROOT` is found, will use `I_MPI_ROOT` instead. Not used when using `NO_DIST=1`
* ``NO_DIST``: set to '1', 'yes' or alike to build without support for distributed mode
* ``NO_STREAM``: set to '1', 'yes' or alike to build without support for streaming mode
* ``NO_DPC``: set to '1', 'yes' or alike to build without support of oneDAL DPC++ interfaces
* ``OFF_ONEDAL_IFACE``: set to '1' to build without the support of oneDAL interfaces
* ``MAKEFLAGS``: the last `-j` flag determines the number of threads for building the onedal extension. It will default to the number of CPU threads when not set.

**Note:** in order to use distributed mode, `mpi4py` is also required, and needs to be built with the same MPI backend as scikit-learn-intelex.
**Note:** The `-j` flag in the ``MAKEFLAGS`` environment variable is superseded in `setup.py` modes which support the ``--parallel`` and `-j` command line flags.


### Build Extension for Scikit-learn

- To install the package:

   ```bash
   cd <checkout-dir>
   python setup.py install
   ```

- To install the package in the development mode:

   ```bash
   cd <checkout-dir>
   python setup.py develop
   ```

- To install scikit-learn-intelex without checking for dependencies:

   ```bash
   cd <checkout-dir>
   python setup.py install --single-version-externally-managed --record=record.txt
   ```
   ```bash
   cd <checkout-dir>
   python setup.py develop --no-deps
   ```

Where: 

* Keys `--single-version-externally-managed` and `--no-deps` are required to not download daal4py after the installation of Extension for Scikit-learn. 
* The `develop` mode does not install the package but creates a `.egg-link` in the deployment directory
back to the project source-code directory. That way, you can edit the source code and see the changes
without reinstalling the package after a small change.
* `--single-version-externally-managed` is an option for Python packages instructing the setuptools module to create a package that the host's package manager can easily manage.

- To build the python module without installing it:

   ```bash
   cd <checkout-dir>
   python setup.py build_ext --inplace --force
   python setup.py build
   ```

**Note1:** the `daal4py` extension module which is built through `build_ext` does not use any kind of build caching for incremental compilation. For development purposes, one might want to use it together with `ccache`, for example by setting `export CXX="ccache icpx"`.

**Note2:** the `setup.py` file will accept an optional argument `--abs-rpath` on linux (for all of `build`/`install`/`develop`/etc.) which will make it add the absolute path to oneDAL's shared objects (.so files) to the rpath of the scikit-learn-intelex extension's shared object files in order to load them automatically. This is not necessary when installing from pip or conda, but can be helpful for development purposes when using a from-source build of oneDAL that resides in a custom folder, as it won't assume that oneDAL's files will be found under default system paths. Example:

```shell
python setup.py build_ext --inplace --force --abs-rpath
python setup.py build --abs-rpath
```

**Note:** when building `scikit-learn-intelex` from source with this option, it will use the oneDAL library with which it was compiled. oneDAL has dependencies on other libraries such as TBB, which is also distributed as a python package through `pip` and as a `conda` package. By default, a conda environment will first try to load TBB from its own packages if it is installed in the environment, which might cause issues if oneDAL was compiled with a system TBB instead of a conda one. In such cases, it is advised to either uninstall TBB from pip/conda (it will be loaded from the oneDAL library which links to it), or modify the order of search paths in environment variables like `${LD_LIBRARY_PATH}`.

### Debug Builds

To build modules with debugging symbols and assertions enabled, pass argument `--debug` to the setup command - e.g.:

```shell
python setup.py build_ext --inplace --force --abs-rpath --debug
python setup.py build --abs-rpath --debug
```

_**Note:** on Windows, this will only add debugging symbols for the `onedal` extension modules, but not for the `daal4py` extension module._

### Building with ASAN

In order to use AddressSanitizer (ASan) together with `scikit-learn-intelex`, it's necessary to:
* Build both oneDAL and scikit-learn-intelex with ASan and with debug symbols (otherwise error traces will not be very informative).
* Preload the ASan runtime when executing the Python process that imports `scikit-learn-intelex`.
* Optionally, configure Python to use `malloc` as default allocator to reduce the number of false-positive leak reports.

See the instructions on the oneDAL repository for building the library from source with ASAN enabled:
https://github.com/uxlfoundation/oneDAL/blob/main/INSTALL.md

When building `scikit-learn-intelex`, the system's default compiler is used unless specified otherwise through variables such as `$CXX`. In order to avoid issues with incompatible runtimes of ASan, one might want to change the compiler to ICX if oneDAL was built with ICX (the default for it).

The compiler and flags to build with both ASan and debug symbols can be controlled through environment variables - **assuming a Linux system** (ASan on Windows has not been tested):
```shell
export CC="icx -fsanitize=address -g"
export CXX="icpx -fsanitize=address -g"
```

_Hint: the Cython module `daal4py` that gets built through `build_ext` does not do incremental compilation, so one might want to add `ccache` into the compiler call for development purposes - e.g. `CXX="ccache icx  -fsanitize=address -g"`._

The ASan runtime used by ICX is the same as the one by Clang. It's possible to preload the ASan runtime for GNU if that's the system's default through e.g. `LD_PRELOAD=libasan.so` or similar. However, one might need to specifically pass the paths from Clang to get the same ASan runtime as for oneDAL if that is not the system's default compiler:
```shell
export LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan-x86_64.so)"
```

_Note: this requires both `clang` and its runtime libraries to be installed. If using toolkits from `conda-forge`, then using `libclang_rt` requires installing package `compiler-rt`, in addition to `clang` and `clangxx`._

Then, the Python memory allocator can be set to `malloc` like this:
```shell
export PYTHONMALLOC=malloc
```

Putting it all together, the earlier examples building the library in-place and executing a python file with it become as follows:
```shell
source <path to ASan-enabled oneDAL env.sh>
CC="ccache icx -fsanitize=address -g" CXX="ccache icpx -fsanitize=address -g" python setup.py build_ext --inplace --force --abs-rpath
CC="icx -fsanitize=address -g" CXX="icpx -fsanitize=address -g" python setup.py build --abs-rpath
LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan-x86_64.so)" PYTHONMALLOC=malloc PYTHONPATH=$(pwd) python <python file.py>
```

_Be aware that ASan is known to generate many false-positive reports of memory leaks when used with oneDAL, NumPy, and SciPy._

## Build from Sources with `conda-build`

Extension for Scikit-learn* is easily built from the sources using only one command and `conda-build` utility. 

### Prerequisites for `conda-build`

* any `conda` distribution (`miniforge` is recommended)
* `conda-build` and `conda-verify` installed in a conda environment
* (Windows only) Microsoft Visual Studio*
* (optional) Intel(R) oneAPI DPC++/C++ Compiler

`conda-build` config requires **2022** version of Microsoft Visual Studio* by default, you can specify another version in `conda-recipe/conda_build_config.yaml` if needed.

In order to enable DPC++ interfaces support on Windows, you need to set `DPCPPROOT` environment variable pointing to DPC++/C++ Compiler distribution.
Conda-forge distribution of DPC++ compiler is used by default on Linux, but you still can set your own distribution via `DPCPPROOT` variable.

### Build Extension for Scikit-learn with `conda-build`

Create and verify `scikit-learn-intelex` conda package with next command executed from root of sklearnex repo:

```bash
conda build .
```

## Next Steps

- [Learn what patching is and how to patch scikit-learn](https://uxlfoundation.github.io/scikit-learn-intelex/latest/what-is-patching.html)
- [Start using scikit-learn-intelex](https://uxlfoundation.github.io/scikit-learn-intelex/latest/quick-start.html)
