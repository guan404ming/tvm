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

"""WebGPU backend modules for TVM Relax.

This module provides WebGPU-specific backends and optimizations,
including integration with the WebInfer kernel library.
"""

from .webinfer import (
    # Module generation functions (similar to FlashInfer pattern)
    gen_webinfer_prefill_module,
    gen_webinfer_decode_module,
    gen_webinfer_ragged_prefill_module,
    gen_webinfer_attention_module,
    gen_webinfer_norm_module,
    gen_webinfer_gemm_module,
    # ExternFunc call generators
    create_webinfer_call,
    create_webinfer_prefill_call,
    create_webinfer_decode_call,
    create_webinfer_ragged_prefill_call,
    # Configuration helpers
    get_webinfer_kernel_configs,
)

__all__ = [
    # Module generation
    "gen_webinfer_prefill_module",
    "gen_webinfer_decode_module",
    "gen_webinfer_ragged_prefill_module",
    "gen_webinfer_attention_module",
    "gen_webinfer_norm_module",
    "gen_webinfer_gemm_module",
    # Call generators
    "create_webinfer_call",
    "create_webinfer_prefill_call",
    "create_webinfer_decode_call",
    "create_webinfer_ragged_prefill_call",
    # Configuration
    "get_webinfer_kernel_configs",
]
