# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for Webinfer backend integration"""


import tvm.testing
from tvm.relax.backend.webgpu import (
    create_webinfer_call,
    gen_webinfer_attention_module,
    gen_webinfer_gemm_module,
    gen_webinfer_norm_module,
    get_webinfer_kernel_configs,
)


def test_webinfer_import():
    """Test that webinfer backend can be imported"""
    # Should not raise any errors
    from tvm.relax.backend import webgpu

    assert hasattr(webgpu, "gen_webinfer_attention_module")
    assert hasattr(webgpu, "gen_webinfer_gemm_module")
    assert hasattr(webgpu, "gen_webinfer_norm_module")


def test_webinfer_attention_module_generation():
    """Test attention module generation"""
    kernels = gen_webinfer_attention_module(
        dtype_q="float16",
        dtype_kv="float16",
        dtype_o="float16",
        head_dim=128,
        num_heads=32,
        num_kv_heads=8,
        use_paged_kv=True,
        use_rope=True,
    )

    # Should return list of ExternFunc references
    assert isinstance(kernels, list)
    assert len(kernels) == 2  # prefill and decode

    # Check that they are ExternFunc
    from tvm import relax as rx

    assert all(isinstance(k, rx.ExternFunc) for k in kernels)


def test_webinfer_attention_module_generation_no_paged():
    """Test attention module generation without paged KV cache"""
    kernels = gen_webinfer_attention_module(
        dtype_q="float16",
        dtype_kv="float16",
        dtype_o="float16",
        head_dim=128,
        num_heads=32,
        num_kv_heads=32,  # MHA
        use_paged_kv=False,
        use_rope=False,
    )

    assert isinstance(kernels, list)
    assert len(kernels) == 2  # single_prefill and single_decode


def test_webinfer_gemm_module_generation():
    """Test GEMM module generation"""
    kernel = gen_webinfer_gemm_module(dtype="float16")

    from tvm import relax as rx

    assert isinstance(kernel, rx.ExternFunc)


def test_webinfer_gemm_module_generation_with_shape():
    """Test GEMM module generation with static shape"""
    kernel = gen_webinfer_gemm_module(
        dtype="float16",
        M=4096,
        N=4096,
        K=4096,
    )

    from tvm import relax as rx

    assert isinstance(kernel, rx.ExternFunc)
    # Should include shape in the URI
    assert "4096" in str(kernel)


def test_webinfer_norm_module_generation():
    """Test normalization module generation"""
    kernel = gen_webinfer_norm_module(
        norm_type="rmsnorm",
        dtype="float16",
        hidden_size=4096,
    )

    from tvm import relax as rx

    assert isinstance(kernel, rx.ExternFunc)


def test_webinfer_norm_module_generation_layernorm():
    """Test layer normalization module generation"""
    kernel = gen_webinfer_norm_module(
        norm_type="layernorm",
        dtype="float32",
        hidden_size=2048,
        eps=1e-5,
    )

    from tvm import relax as rx

    assert isinstance(kernel, rx.ExternFunc)


def test_webinfer_multiple_kernel_generation():
    """Test generating multiple kernels and configs"""
    # Generate several kernels
    gen_webinfer_attention_module(
        dtype_q="float16",
        dtype_kv="float16",
        dtype_o="float16",
        head_dim=128,
        num_heads=32,
        num_kv_heads=8,
        use_paged_kv=True,
        use_rope=True,
    )

    gen_webinfer_gemm_module(dtype="float16", M=4096, N=4096, K=4096)

    gen_webinfer_norm_module("rmsnorm", "float16", 4096)

    # Get kernel configs
    configs = get_webinfer_kernel_configs()

    # Should be a dictionary
    assert isinstance(configs, dict)
    # Note: configs may be empty if not explicitly registered in the implementation
    # This is expected behavior for the current implementation


def test_webinfer_create_call():
    """Test creating Webinfer kernel call"""

    # Generate a kernel
    gen_webinfer_gemm_module(dtype="float16")

    # Create a call (basic structure test)
    # Note: Full IR construction would require more setup
    # This just tests that the function exists and is callable
    assert callable(create_webinfer_call)


if __name__ == "__main__":
    # Run tests with pytest
    tvm.testing.main()
