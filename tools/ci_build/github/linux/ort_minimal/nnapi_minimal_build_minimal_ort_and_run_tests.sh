#!/bin/bash

# This script takes one command line argument, the root of the current ONNX Runtime project

set -e
set -x

ORT_ROOT=$1
MIN_BUILD_DIR=$ORT_ROOT/build_nnapi_minimal

# Remove builds from previous CPU and NNAPI full Android ORT build to free up disk space
rm -rf $ORT_ROOT/build
rm -rf $ORT_ROOT/build_nnapi

# Build with reduced ops requires onnx
# python3 -m pip install -U --user onnx

# generate config for ops and types used in the .ort models in the testdata dir
python3 $ORT_ROOT/tools/python/create_reduced_build_config --format ORT --enable_type_reduction \
    $ORT_ROOT/onnxruntime/test/testdata \
    $ORT_ROOT/onnxruntime/test/testdata/required_ops_and_types.config

# Build minimal package for Android x86_64 Emulator
# No test will be triggered in the build process
# UT will be triggered separately below
python3 $ORT_ROOT/tools/ci_build/build.py \
    --build_dir $MIN_BUILD_DIR \
    --config Debug \
    --skip_submodule_sync \
    --parallel \
    --cmake_generator=Ninja \
    --use_nnapi \
    --android \
    --android_sdk_path $ANDROID_HOME \
    --android_ndk_path $ANDROID_HOME/ndk-bundle \
    --android_abi=x86_64 \
    --android_api=29 \
    --minimal_build extended \
    --disable_rtti \
    --disable_ml_ops \
    --disable_exceptions \
    --include_ops_by_config $ORT_ROOT/onnxruntime/test/testdata/required_ops_and_types.config \
    --skip_tests

# Push onnxruntime_test_all and testdata to emulator
adb push $MIN_BUILD_DIR/Debug/onnxruntime_test_all /data/local/tmp/
adb push $MIN_BUILD_DIR/Debug/testdata /data/local/tmp/

# Perform the UT
adb shell 'cd /data/local/tmp/ && ./onnxruntime_test_all'