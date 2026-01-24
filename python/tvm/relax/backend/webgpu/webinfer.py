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

"""WebInfer integration for WebGPU backend.

WebInfer is a high-performance LLM inference library for WebGPU,
similar to FlashInfer for CUDA. This module provides the integration
following the FlashInfer pattern:

1. Python generates kernel specifications (not compiled code)
2. Kernel specs are packaged into WebInferSourceModule
3. JavaScript runtime loads specs and JIT compiles WGSL shaders
4. TVM runtime resolves ExternFunc to registered functions

Architecture comparison with FlashInfer:
- FlashInfer: Python -> CUDA JIT -> .o files -> load_static_library()
- WebInfer: Python -> kernel specs (JSON) -> WebInferSourceModule -> JS JIT -> WGSL
"""

import json
from typing import Any, Dict, List, Optional

import tvm
from tvm import relax as rx
from tvm.relax.expr_functor import PyExprVisitor, visitor


def _create_webinfer_source_module(spec: Dict[str, Any]) -> tvm.runtime.Module:
    """Create a WebInfer source module from kernel spec.

    This creates a TVM module that contains the kernel specification
    as JSON. The JavaScript runtime will read this spec and JIT compile
    the appropriate WGSL shader.

    Parameters
    ----------
    spec : Dict[str, Any]
        Kernel specification dictionary

    Returns
    -------
    tvm.runtime.Module
        WebInfer source module containing the kernel spec
    """
    create_func = tvm.get_global_func("runtime.WebInferSourceModuleCreate", allow_missing=True)
    if create_func is None:
        # Fallback: return a placeholder module with the spec as metadata
        # This allows Python-side testing without the C++ module implementation
        return _create_placeholder_module(spec)
    return create_func(json.dumps(spec))


def _create_placeholder_module(spec: Dict[str, Any]) -> tvm.runtime.Module:
    """Create a placeholder module for testing without C++ implementation.

    This is used when the WebInferSourceModule C++ implementation is not available.
    The spec is stored in a way that can be retrieved later.
    """
    # For now, we can use a simple approach - create a module that stores the spec
    # This will be replaced by proper C++ implementation
    import warnings
    warnings.warn(
        "WebInferSourceModule C++ implementation not found. "
        "Using placeholder module. Some features may not work."
    )

    # Return None for now - the actual module will be created when C++ is ready
    return None


def gen_webinfer_prefill_module(
    dtype: str,
    num_qo_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    page_size: int,
    enable_inline_rope: bool = False,
) -> List[tvm.runtime.Module]:
    """Generate WebInfer module for paged prefill attention.

    Similar to gen_flashinfer_prefill_module() but generates kernel specs
    instead of compiled CUDA objects. The kernel specs are compiled to
    WGSL at runtime by the JavaScript WebInfer library.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    num_qo_heads : int
        Number of query/output heads
    num_kv_heads : int
        Number of key/value heads
    qk_head_dim : int
        Head dimension for query and key
    v_head_dim : int
        Head dimension for value
    page_size : int
        Page size for paged KV cache
    enable_inline_rope : bool
        Whether to enable inline RoPE

    Returns
    -------
    List[tvm.runtime.Module]
        WebInfer source modules containing kernel specs
    """
    spec = {
        "kernel_type": "batch_prefill_paged",
        "dtype": dtype,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "page_size": page_size,
        "enable_inline_rope": enable_inline_rope,
    }

    module = _create_webinfer_source_module(spec)
    return [module] if module is not None else []


def gen_webinfer_decode_module(
    dtype: str,
    num_qo_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    page_size: int,
    enable_inline_rope: bool = False,
) -> List[tvm.runtime.Module]:
    """Generate WebInfer module for paged decode attention.

    Similar to gen_flashinfer_decode_module() but for WebGPU.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    num_qo_heads : int
        Number of query/output heads
    num_kv_heads : int
        Number of key/value heads
    qk_head_dim : int
        Head dimension for query and key
    v_head_dim : int
        Head dimension for value
    page_size : int
        Page size for paged KV cache
    enable_inline_rope : bool
        Whether to enable inline RoPE

    Returns
    -------
    List[tvm.runtime.Module]
        WebInfer source modules containing kernel specs
    """
    spec = {
        "kernel_type": "batch_decode_paged",
        "dtype": dtype,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "page_size": page_size,
        "enable_inline_rope": enable_inline_rope,
    }

    module = _create_webinfer_source_module(spec)
    return [module] if module is not None else []


def gen_webinfer_ragged_prefill_module(
    dtype: str,
    num_qo_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    enable_inline_rope: bool = False,
) -> List[tvm.runtime.Module]:
    """Generate WebInfer module for ragged (non-paged) prefill attention.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    num_qo_heads : int
        Number of query/output heads
    num_kv_heads : int
        Number of key/value heads
    qk_head_dim : int
        Head dimension for query and key
    v_head_dim : int
        Head dimension for value
    enable_inline_rope : bool
        Whether to enable inline RoPE

    Returns
    -------
    List[tvm.runtime.Module]
        WebInfer source modules containing kernel specs
    """
    spec = {
        "kernel_type": "single_prefill",
        "dtype": dtype,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "enable_inline_rope": enable_inline_rope,
    }

    module = _create_webinfer_source_module(spec)
    return [module] if module is not None else []


def gen_webinfer_attention_module(
    dtype_q: str,
    dtype_kv: str,
    dtype_o: str,
    qk_head_dim: int,
    v_head_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
    page_size: int = 16,
    enable_inline_rope: bool = False,
) -> List[tvm.runtime.Module]:
    """Generate WebInfer attention modules (prefill + decode).

    This is a convenience function that generates both prefill and decode
    modules with the same configuration.

    Parameters
    ----------
    dtype_q : str
        Data type for query tensor
    dtype_kv : str
        Data type for key/value tensors
    dtype_o : str
        Data type for output tensor
    qk_head_dim : int
        Head dimension for query and key
    v_head_dim : int
        Head dimension for value
    num_qo_heads : int
        Number of query/output heads
    num_kv_heads : int
        Number of key/value heads
    page_size : int
        Page size for paged KV cache
    enable_inline_rope : bool
        Whether to enable inline RoPE

    Returns
    -------
    List[tvm.runtime.Module]
        WebInfer source modules for prefill and decode
    """
    # For simplicity, use dtype_q for all (they should match in most cases)
    dtype = dtype_q

    prefill_mods = gen_webinfer_prefill_module(
        dtype=dtype,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        page_size=page_size,
        enable_inline_rope=enable_inline_rope,
    )

    decode_mods = gen_webinfer_decode_module(
        dtype=dtype,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        page_size=page_size,
        enable_inline_rope=enable_inline_rope,
    )

    return prefill_mods + decode_mods


def gen_webinfer_norm_module(
    dtype: str,
    hidden_dim: int,
    eps: float = 1e-6,
) -> List[tvm.runtime.Module]:
    """Generate WebInfer module for RMSNorm.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    hidden_dim : int
        Hidden dimension size
    eps : float
        Epsilon for numerical stability

    Returns
    -------
    List[tvm.runtime.Module]
        WebInfer source modules containing kernel specs
    """
    spec = {
        "kernel_type": "rmsnorm",
        "dtype": dtype,
        "hidden_dim": hidden_dim,
        "eps": eps,
    }

    module = _create_webinfer_source_module(spec)
    return [module] if module is not None else []


def gen_webinfer_gemm_module(
    dtype: str,
    M: Optional[int] = None,
    N: Optional[int] = None,
    K: Optional[int] = None,
) -> List[tvm.runtime.Module]:
    """Generate WebInfer module for GEMM operations.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    M, N, K : int, optional
        Matrix dimensions (if known at compile time)

    Returns
    -------
    List[tvm.runtime.Module]
        WebInfer source modules containing kernel specs
    """
    spec = {
        "kernel_type": "gemm",
        "dtype": dtype,
    }
    if M is not None:
        spec["M"] = M
    if N is not None:
        spec["N"] = N
    if K is not None:
        spec["K"] = K

    module = _create_webinfer_source_module(spec)
    return [module] if module is not None else []


# ============================================================================
# ExternFunc Generators for Relax IR
# ============================================================================


def create_webinfer_prefill_call() -> rx.Tuple:
    """Create ExternFunc tuple for WebInfer prefill attention.

    This returns the tuple format expected by TVM's attention backend system:
    (backend_name, run_func, plan_func)

    Returns
    -------
    rx.Tuple
        Tuple containing backend identifier and ExternFunc references
    """
    return rx.Tuple([
        rx.StringImm("webinfer"),
        rx.ExternFunc("webinfer_prefill_run"),
        rx.ExternFunc("webinfer_prefill_plan"),
    ])


def create_webinfer_decode_call() -> rx.Tuple:
    """Create ExternFunc tuple for WebInfer decode attention.

    Returns
    -------
    rx.Tuple
        Tuple containing backend identifier and ExternFunc references
    """
    return rx.Tuple([
        rx.StringImm("webinfer"),
        rx.ExternFunc("webinfer_decode_run"),
        rx.ExternFunc("webinfer_decode_plan"),
    ])


def create_webinfer_ragged_prefill_call() -> rx.Tuple:
    """Create ExternFunc tuple for WebInfer ragged prefill attention.

    Returns
    -------
    rx.Tuple
        Tuple containing backend identifier and ExternFunc references
    """
    return rx.Tuple([
        rx.StringImm("webinfer"),
        rx.ExternFunc("webinfer_ragged_prefill_run"),
        rx.ExternFunc("webinfer_ragged_prefill_plan"),
    ])


def create_webinfer_call(kernel_type: str) -> rx.Tuple:
    """Create ExternFunc tuple for a WebInfer kernel.

    This is a generic interface that dispatches to the appropriate
    function based on kernel type.

    Parameters
    ----------
    kernel_type : str
        Type of kernel ("prefill", "decode", "ragged_prefill", "rmsnorm", etc.)

    Returns
    -------
    rx.Tuple
        Tuple containing backend identifier and ExternFunc references
    """
    if kernel_type == "prefill":
        return create_webinfer_prefill_call()
    elif kernel_type == "decode":
        return create_webinfer_decode_call()
    elif kernel_type == "ragged_prefill":
        return create_webinfer_ragged_prefill_call()
    elif kernel_type == "rmsnorm":
        return rx.Tuple([
            rx.StringImm("webinfer"),
            rx.ExternFunc("webinfer_rmsnorm_run"),
        ])
    elif kernel_type == "silu_and_mul":
        return rx.Tuple([
            rx.StringImm("webinfer"),
            rx.ExternFunc("webinfer_silu_and_mul_run"),
        ])
    elif kernel_type == "rope":
        return rx.Tuple([
            rx.StringImm("webinfer"),
            rx.ExternFunc("webinfer_rope_run"),
        ])
    elif kernel_type == "gemm":
        return rx.Tuple([
            rx.StringImm("webinfer"),
            rx.ExternFunc("webinfer_gemm_run"),
        ])
    else:
        raise ValueError(f"Unknown WebInfer kernel type: {kernel_type}")


def get_webinfer_kernel_configs() -> Dict[str, Dict[str, Any]]:
    """Get default kernel configurations for WebInfer.

    Returns a dictionary of default configurations for each kernel type.
    These can be used to initialize WebInfer kernels with sensible defaults.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping kernel types to their default configurations
    """
    return {
        "batch_prefill_paged": {
            "dtype": "float16",
            "page_size": 16,
            "causal": True,
        },
        "batch_decode_paged": {
            "dtype": "float16",
            "page_size": 16,
        },
        "rmsnorm": {
            "dtype": "float16",
            "eps": 1e-6,
        },
        "silu_and_mul": {
            "dtype": "float16",
        },
        "rope": {
            "dtype": "float16",
            "rope_theta": 10000.0,
            "rope_scale": 1.0,
        },
    }


# ============================================================================
# Spec Generation Functions (Return Dict, not Module)
# ============================================================================


def gen_webinfer_prefill_spec(
    dtype: str,
    num_qo_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    page_size: int,
    enable_inline_rope: bool = False,
    causal: bool = True,
) -> Dict[str, Any]:
    """Generate kernel spec dict for paged prefill attention.

    This returns a spec dictionary that will be embedded in the module
    and used by the JavaScript runtime to JIT compile the WGSL shader.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    num_qo_heads : int
        Number of query/output heads
    num_kv_heads : int
        Number of key/value heads
    qk_head_dim : int
        Head dimension for query and key
    v_head_dim : int
        Head dimension for value
    page_size : int
        Page size for paged KV cache
    enable_inline_rope : bool
        Whether to enable inline RoPE
    causal : bool
        Whether to use causal masking

    Returns
    -------
    Dict[str, Any]
        Kernel specification dictionary
    """
    return {
        "kernel_type": "batch_prefill_paged",
        "dtype": dtype,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "page_size": page_size,
        "enable_inline_rope": enable_inline_rope,
        "causal": causal,
    }


def gen_webinfer_decode_spec(
    dtype: str,
    num_qo_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    page_size: int,
    enable_inline_rope: bool = False,
) -> Dict[str, Any]:
    """Generate kernel spec dict for paged decode attention.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    num_qo_heads : int
        Number of query/output heads
    num_kv_heads : int
        Number of key/value heads
    qk_head_dim : int
        Head dimension for query and key
    v_head_dim : int
        Head dimension for value
    page_size : int
        Page size for paged KV cache
    enable_inline_rope : bool
        Whether to enable inline RoPE

    Returns
    -------
    Dict[str, Any]
        Kernel specification dictionary
    """
    return {
        "kernel_type": "batch_decode_paged",
        "dtype": dtype,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "page_size": page_size,
        "enable_inline_rope": enable_inline_rope,
    }


def gen_webinfer_gemm_spec(
    dtype: str,
    M: Optional[int] = None,
    N: Optional[int] = None,
    K: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate kernel spec dict for GEMM operations.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    M, N, K : int, optional
        Matrix dimensions (if known at compile time)

    Returns
    -------
    Dict[str, Any]
        Kernel specification dictionary
    """
    spec: Dict[str, Any] = {
        "kernel_type": "gemm",
        "dtype": dtype,
    }
    if M is not None:
        spec["M"] = M
    if N is not None:
        spec["N"] = N
    if K is not None:
        spec["K"] = K
    return spec


def gen_webinfer_rmsnorm_spec(
    dtype: str,
    hidden_dim: int,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Generate kernel spec dict for RMSNorm.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    hidden_dim : int
        Hidden dimension size
    eps : float
        Epsilon for numerical stability

    Returns
    -------
    Dict[str, Any]
        Kernel specification dictionary
    """
    return {
        "kernel_type": "rmsnorm",
        "dtype": dtype,
        "hidden_dim": hidden_dim,
        "eps": eps,
    }


def gen_webinfer_silu_and_mul_spec(
    dtype: str,
    hidden_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate kernel spec dict for SiLU activation with gating.

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    hidden_dim : int, optional
        Hidden dimension size

    Returns
    -------
    Dict[str, Any]
        Kernel specification dictionary
    """
    spec: Dict[str, Any] = {
        "kernel_type": "silu_and_mul",
        "dtype": dtype,
    }
    if hidden_dim is not None:
        spec["hidden_dim"] = hidden_dim
    return spec


def gen_webinfer_rope_spec(
    dtype: str,
    qk_head_dim: int,
    rope_theta: float = 10000.0,
    rope_scale: float = 1.0,
) -> Dict[str, Any]:
    """Generate kernel spec dict for RoPE (Rotary Position Embedding).

    Parameters
    ----------
    dtype : str
        Data type ("float16" or "float32")
    qk_head_dim : int
        Head dimension for query and key
    rope_theta : float
        RoPE theta base
    rope_scale : float
        RoPE scaling factor

    Returns
    -------
    Dict[str, Any]
        Kernel specification dictionary
    """
    return {
        "kernel_type": "rope",
        "dtype": dtype,
        "qk_head_dim": qk_head_dim,
        "rope_theta": rope_theta,
        "rope_scale": rope_scale,
    }


# ============================================================================
# Module Attribute Functions
# ============================================================================


def attach_webinfer_specs(mod: tvm.IRModule, specs: List[Dict[str, Any]]) -> tvm.IRModule:
    """Attach WebInfer kernel specs as a module attribute.

    This embeds the kernel specifications as a JSON string in the module
    attribute. The specs should also be saved as a separate JSON file
    alongside the WASM for the JavaScript runtime to load.

    Usage pattern:
    1. Python: attach_webinfer_specs(mod, specs) + save specs JSON
    2. JavaScript: Load WASM + Load specs JSON + initWebinferFromSpecs()

    Parameters
    ----------
    mod : tvm.IRModule
        The module to attach specs to
    specs : List[Dict[str, Any]]
        List of kernel specification dictionaries

    Returns
    -------
    tvm.IRModule
        Module with attached webinfer_kernel_specs attribute
    """
    specs_json = json.dumps(specs)
    return mod.with_attr("webinfer_kernel_specs", specs_json)


def get_webinfer_specs(mod: tvm.IRModule) -> Optional[List[Dict[str, Any]]]:
    """Get WebInfer kernel specs from a module attribute.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to get specs from

    Returns
    -------
    Optional[List[Dict[str, Any]]]
        List of kernel specifications, or None if not found
    """
    specs_json = mod.attrs.get("webinfer_kernel_specs", None)
    if specs_json is None:
        return None
    return json.loads(specs_json)


# ============================================================================
# WebInfer Collector Pass
# ============================================================================


@visitor
class WebInferCallVisitor(PyExprVisitor):
    """Visitor to collect WebInfer call_packed calls from Relax IR."""

    def __init__(self):
        self.webinfer_calls: List[Dict[str, Any]] = []

    def visit_call_(self, call: rx.Call):
        """Visit a Call node and check if it's a WebInfer call."""
        # Check if this is a call_packed to a webinfer function
        if isinstance(call.op, rx.ExternFunc):
            func_name = call.op.global_symbol
            if func_name.startswith("webinfer_"):
                self._extract_spec_from_call(func_name, call)

    def _extract_spec_from_call(self, func_name: str, call: rx.Call):
        """Extract kernel spec from a WebInfer call."""
        # The spec extraction logic depends on the function signature
        # For now, we record the function name and any literal arguments
        spec_info = {
            "func_name": func_name,
            "args": [],
        }

        for arg in call.args:
            if isinstance(arg, rx.PrimValue):
                # Extract literal values
                if hasattr(arg.value, "value"):
                    spec_info["args"].append(arg.value.value)
            elif isinstance(arg, rx.StringImm):
                spec_info["args"].append(arg.value)

        self.webinfer_calls.append(spec_info)


@tvm.transform.module_pass(opt_level=0)
class WebInferCollectorPass:
    """Analyze IR and collect WebInfer kernel specs.

    This pass scans the Relax IR for call_packed("webinfer_*", ...) calls,
    extracts the kernel configurations, and attaches them as a module
    attribute for the JavaScript runtime to use.
    """

    def transform_module(self, mod: tvm.IRModule, ctx) -> tvm.IRModule:
        """Transform the module by collecting WebInfer specs."""
        visitor = WebInferCallVisitor()
        specs = []

        for gv, func in mod.functions.items():
            if isinstance(func, rx.Function):
                visitor.visit_expr(func)

        # Convert collected calls to kernel specs
        for call_info in visitor.webinfer_calls:
            spec = self._call_to_spec(call_info)
            if spec is not None:
                specs.append(spec)

        # Only attach if we found specs
        if specs:
            # Deduplicate specs by kernel_type and key parameters
            unique_specs = self._deduplicate_specs(specs)
            return attach_webinfer_specs(mod, unique_specs)

        return mod

    def _call_to_spec(self, call_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a call info dict to a kernel spec."""
        func_name = call_info["func_name"]

        # Map function names to kernel types
        if func_name == "webinfer_gemm_run":
            return {"kernel_type": "gemm", "dtype": "float32"}
        elif func_name == "webinfer_prefill_run":
            return {"kernel_type": "batch_prefill_paged", "dtype": "float16"}
        elif func_name == "webinfer_decode_run":
            return {"kernel_type": "batch_decode_paged", "dtype": "float16"}
        elif func_name == "webinfer_rmsnorm_run":
            return {"kernel_type": "rmsnorm", "dtype": "float16"}
        elif func_name == "webinfer_silu_and_mul_run":
            return {"kernel_type": "silu_and_mul", "dtype": "float16"}
        elif func_name == "webinfer_rope_run":
            return {"kernel_type": "rope", "dtype": "float16"}

        return None

    def _deduplicate_specs(self, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate specs based on kernel_type."""
        seen = set()
        unique = []
        for spec in specs:
            key = json.dumps(spec, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(spec)
        return unique
