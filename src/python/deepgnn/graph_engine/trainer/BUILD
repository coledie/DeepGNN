# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "deepgnn_trainer",
    srcs = [
        "__init__.py",
        "args.py",
        "factory.py",
    ],
    deps = [
        "//src/python/deepgnn/pytorch/modeling:deepgnn_pytorch_modeling",
    ],
    visibility = ["//visibility:public"],
)
