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
"""Unit tests for WebInfer module generation functions.

These tests verify that the Python-side WebInfer integration works correctly,
including kernel spec generation and ExternFunc creation. These tests don't
require WebGPU hardware and can run on any platform.
"""

import json
import pytest

import tvm
from tvm import relax as rx


class TestWebInferModuleGeneration:
    """Test WebInfer module generation functions."""

    def test_gen_webinfer_prefill_module(self):
        """Test prefill module generation."""
        from tvm.relax.backend.webgpu import webinfer

        modules = webinfer.gen_webinfer_prefill_module(
            dtype="float16",
            num_qo_heads=32,
            num_kv_heads=8,
            qk_head_dim=128,
            v_head_dim=128,
            page_size=16,
            enable_inline_rope=False,
        )

        # Should return a list (possibly empty if C++ module not available)
        assert isinstance(modules, list)

    def test_gen_webinfer_decode_module(self):
        """Test decode module generation."""
        from tvm.relax.backend.webgpu import webinfer

        modules = webinfer.gen_webinfer_decode_module(
            dtype="float16",
            num_qo_heads=32,
            num_kv_heads=8,
            qk_head_dim=128,
            v_head_dim=128,
            page_size=16,
            enable_inline_rope=False,
        )

        assert isinstance(modules, list)

    def test_gen_webinfer_ragged_prefill_module(self):
        """Test ragged prefill module generation."""
        from tvm.relax.backend.webgpu import webinfer

        modules = webinfer.gen_webinfer_ragged_prefill_module(
            dtype="float16",
            num_qo_heads=32,
            num_kv_heads=8,
            qk_head_dim=128,
            v_head_dim=128,
            enable_inline_rope=False,
        )

        assert isinstance(modules, list)

    def test_gen_webinfer_attention_module(self):
        """Test combined attention module generation (prefill + decode)."""
        from tvm.relax.backend.webgpu import webinfer

        modules = webinfer.gen_webinfer_attention_module(
            dtype_q="float16",
            dtype_kv="float16",
            dtype_o="float16",
            qk_head_dim=128,
            v_head_dim=128,
            num_qo_heads=32,
            num_kv_heads=8,
            page_size=16,
            enable_inline_rope=False,
        )

        assert isinstance(modules, list)

    def test_gen_webinfer_norm_module(self):
        """Test RMSNorm module generation."""
        from tvm.relax.backend.webgpu import webinfer

        modules = webinfer.gen_webinfer_norm_module(
            dtype="float16",
            hidden_dim=4096,
            eps=1e-6,
        )

        assert isinstance(modules, list)

    def test_gen_webinfer_gemm_module(self):
        """Test GEMM module generation."""
        from tvm.relax.backend.webgpu import webinfer

        modules = webinfer.gen_webinfer_gemm_module(
            dtype="float16",
            M=512,
            N=4096,
            K=4096,
        )

        assert isinstance(modules, list)


class TestWebInferExternFuncCreation:
    """Test ExternFunc tuple creation for WebInfer."""

    def test_create_webinfer_prefill_call(self):
        """Test prefill ExternFunc tuple creation."""
        from tvm.relax.backend.webgpu import webinfer

        call = webinfer.create_webinfer_prefill_call()

        assert isinstance(call, rx.Tuple)
        # Should have 3 elements: backend_name, run_func, plan_func
        assert len(call.fields) == 3
        assert isinstance(call.fields[0], rx.StringImm)
        assert call.fields[0].value == "webinfer"
        assert isinstance(call.fields[1], rx.ExternFunc)
        assert isinstance(call.fields[2], rx.ExternFunc)

    def test_create_webinfer_decode_call(self):
        """Test decode ExternFunc tuple creation."""
        from tvm.relax.backend.webgpu import webinfer

        call = webinfer.create_webinfer_decode_call()

        assert isinstance(call, rx.Tuple)
        assert len(call.fields) == 3
        assert call.fields[0].value == "webinfer"

    def test_create_webinfer_ragged_prefill_call(self):
        """Test ragged prefill ExternFunc tuple creation."""
        from tvm.relax.backend.webgpu import webinfer

        call = webinfer.create_webinfer_ragged_prefill_call()

        assert isinstance(call, rx.Tuple)
        assert len(call.fields) == 3
        assert call.fields[0].value == "webinfer"

    def test_create_webinfer_call_dispatch(self):
        """Test generic create_webinfer_call dispatcher."""
        from tvm.relax.backend.webgpu import webinfer

        # Test all supported kernel types
        for kernel_type in ["prefill", "decode", "ragged_prefill", "rmsnorm", "silu_and_mul", "rope", "gemm"]:
            call = webinfer.create_webinfer_call(kernel_type)
            assert isinstance(call, rx.Tuple)
            assert call.fields[0].value == "webinfer"

    def test_create_webinfer_call_unknown_type(self):
        """Test that unknown kernel type raises ValueError."""
        from tvm.relax.backend.webgpu import webinfer

        with pytest.raises(ValueError, match="Unknown WebInfer kernel type"):
            webinfer.create_webinfer_call("unknown_kernel")


class TestWebInferKernelConfigs:
    """Test kernel configuration helpers."""

    def test_get_webinfer_kernel_configs(self):
        """Test default kernel configurations."""
        from tvm.relax.backend.webgpu import webinfer

        configs = webinfer.get_webinfer_kernel_configs()

        assert isinstance(configs, dict)
        assert "batch_prefill_paged" in configs
        assert "batch_decode_paged" in configs
        assert "rmsnorm" in configs
        assert "silu_and_mul" in configs
        assert "rope" in configs

        # Check prefill config
        prefill_config = configs["batch_prefill_paged"]
        assert prefill_config["dtype"] == "float16"
        assert prefill_config["page_size"] == 16
        assert prefill_config["causal"] == True

        # Check rmsnorm config
        rmsnorm_config = configs["rmsnorm"]
        assert rmsnorm_config["dtype"] == "float16"
        assert rmsnorm_config["eps"] == 1e-6


class TestWebInferImports:
    """Test that all expected symbols are exported."""

    def test_webgpu_module_exports(self):
        """Test that webgpu module exports webinfer functions."""
        from tvm.relax.backend import webgpu

        # Module generation functions
        assert hasattr(webgpu, "gen_webinfer_prefill_module")
        assert hasattr(webgpu, "gen_webinfer_decode_module")
        assert hasattr(webgpu, "gen_webinfer_ragged_prefill_module")
        assert hasattr(webgpu, "gen_webinfer_attention_module")
        assert hasattr(webgpu, "gen_webinfer_norm_module")
        assert hasattr(webgpu, "gen_webinfer_gemm_module")

        # Call generators
        assert hasattr(webgpu, "create_webinfer_call")
        assert hasattr(webgpu, "create_webinfer_prefill_call")
        assert hasattr(webgpu, "create_webinfer_decode_call")
        assert hasattr(webgpu, "create_webinfer_ragged_prefill_call")

        # Configuration helpers
        assert hasattr(webgpu, "get_webinfer_kernel_configs")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
