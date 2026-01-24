/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * WebInfer integration for TVM WebGPU backend
 *
 * This module provides integration between TVM and WebInfer, following
 * the FlashInfer pattern:
 *
 * 1. Python generates kernel specifications (not compiled code)
 * 2. Kernel specs are packaged into WebInferSourceModule
 * 3. JavaScript runtime loads specs and JIT compiles WGSL shaders
 * 4. TVM runtime resolves ExternFunc to registered functions
 *
 * Architecture comparison with FlashInfer:
 * - FlashInfer: Python -> CUDA JIT -> .o files -> load_static_library()
 * - WebInfer: Python -> kernel specs (JSON) -> WebInferSourceModule -> JS JIT -> WGSL
 */

import * as webinfer from "webinfer";
import { WebGPUContext, GPUPointer } from "./webgpu";

// Type imports - these will be resolved at runtime
type Tensor = any;

/**
 * Kernel specification from TVM Python side
 */
interface KernelSpec {
  kernel_type:
    | "batch_prefill_paged"
    | "batch_decode_paged"
    | "single_prefill"
    | "rmsnorm"
    | "silu_and_mul"
    | "gelu_and_mul"
    | "rope"
    | "gemm"
    | "sampling";
  dtype: "float16" | "float32";
  num_qo_heads?: number;
  num_kv_heads?: number;
  qk_head_dim?: number;
  v_head_dim?: number;
  page_size?: number;
  hidden_dim?: number;
  eps?: number;
  enable_inline_rope?: boolean;
  causal?: boolean;
}

/**
 * Plan info returned by plan functions
 */
interface PrefillPlanInfo {
  key: string;
  batchSize: number;
  totalQoLen: number;
  pageSize: number;
  numQoHeads: number;
  numKvHeads: number;
  qkHeadDim: number;
  vHeadDim: number;
  causal: boolean;
}

interface DecodePlanInfo {
  key: string;
  batchSize: number;
  pageSize: number;
  numQoHeads: number;
  numKvHeads: number;
  qkHeadDim: number;
  vHeadDim: number;
}

interface RaggedPrefillPlanInfo {
  key: string;
  batchSize: number;
  totalQoLen: number;
  numQoHeads: number;
  numKvHeads: number;
  qkHeadDim: number;
  vHeadDim: number;
  causal: boolean;
}

/**
 * Compiled kernel pipeline
 */
interface CompiledPipeline {
  spec: KernelSpec;
  prefillWrapper?: any; // webinfer.prefill.BatchPrefillWithPagedKVCacheWrapper
  decodeWrapper?: any; // webinfer.decode.BatchDecodeWithPagedKVCacheWrapper
}

// Global webinfer context (initialized once)
let webinferCtx: webinfer.WebInferContext | null = null;
const compiledPipelines: Map<string, CompiledPipeline> = new Map();

// Kernel registry for pre-compiled kernels from specs
let kernelRegistry: webinfer.CompiledKernelRegistry | null = null;

/**
 * Generate a unique key for a kernel spec
 */
function getSpecKey(spec: KernelSpec): string {
  const parts: string[] = [spec.kernel_type, spec.dtype];
  if (spec.num_qo_heads !== undefined) parts.push(`qo${spec.num_qo_heads}`);
  if (spec.num_kv_heads !== undefined) parts.push(`kv${spec.num_kv_heads}`);
  if (spec.qk_head_dim !== undefined) parts.push(`qkd${spec.qk_head_dim}`);
  if (spec.v_head_dim !== undefined) parts.push(`vd${spec.v_head_dim}`);
  if (spec.page_size !== undefined) parts.push(`ps${spec.page_size}`);
  return parts.join("_");
}

/**
 * Helper to create webinfer Tensor from TVM Tensor
 * This wraps the same GPUBuffer without copying data (zero-copy)
 */
function toWebinferTensor(
  tvmTensor: Tensor,
  webGPUContext: WebGPUContext,
  dtype: "float16" | "float32"
): webinfer.Tensor {
  const ptr: GPUPointer = tvmTensor.getDataPtr();
  const gpuBuffer = webGPUContext.getBufferFromPtr(ptr);
  return webinfer.Tensor.fromBuffer(
    gpuBuffer,
    tvmTensor.shape as number[],
    dtype
  );
}

/**
 * Initialize WebInfer kernels from module specs.
 * Called when loading TVM module with WebInfer backend.
 *
 * @param registerFunc Function to register a global PackedFunc
 * @param webGPUContext The WebGPU context for buffer access
 * @param getKernelSpecs Function to retrieve kernel specs from TVM module
 */
export async function initWebinferFromModule(
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext,
  getKernelSpecs?: () => string
): Promise<void> {
  try {
    // Initialize webinfer context using TVM's GPU device (zero-copy)
    webinferCtx = webinfer.WebInferContext.fromDevice(webGPUContext.device);
  } catch (error) {
    // Fallback: create a new context
    try {
      webinferCtx = await webinfer.WebInferContext.create();
    } catch (e) {
      console.warn("WebInfer initialization failed:", e);
      // Continue anyway - we still need to register the kernel functions
      // They will error at runtime if called without a valid context
    }
  }

  // If kernel specs are provided, compile them
  if (getKernelSpecs) {
    try {
      const specsJson = getKernelSpecs();
      const specs: KernelSpec[] = JSON.parse(specsJson);

      for (const spec of specs) {
        await registerKernelFromSpec(spec, registerFunc, webGPUContext);
      }
    } catch (error) {
      console.warn("Failed to load kernel specs:", error);
    }
  }

  // Register plan/run functions for attention kernels
  registerAttentionKernels(registerFunc, webGPUContext);

  // Register other kernel types
  registerGemmKernels(registerFunc, webGPUContext);
  registerNormKernels(registerFunc, webGPUContext);
  registerActivationKernels(registerFunc, webGPUContext);
  registerRopeKernels(registerFunc, webGPUContext);
}

/**
 * Register attention kernels (prefill, decode, ragged prefill)
 */
function registerAttentionKernels(
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext
): void {
  // ============================================================
  // Paged Prefill Attention
  // ============================================================

  // Plan function - prepares execution metadata
  registerFunc(
    "webinfer_prefill_plan",
    (
      batchSize: number,
      totalQoLen: number,
      pageSize: number,
      numQoHeads: number,
      numKvHeads: number,
      qkHeadDim: number,
      vHeadDim: number,
      causal: boolean
    ): string => {
      const planInfo: PrefillPlanInfo = {
        key: `prefill_${numQoHeads}_${numKvHeads}_${qkHeadDim}_${pageSize}`,
        batchSize,
        totalQoLen,
        pageSize,
        numQoHeads,
        numKvHeads,
        qkHeadDim,
        vHeadDim,
        causal,
      };
      return JSON.stringify(planInfo);
    }
  );

  // Run function - executes the kernel (synchronous - queues GPU commands)
  registerFunc(
    "webinfer_prefill_run",
    (
      planInfoJson: string,
      q: Tensor,
      pages: Tensor,
      pageIndptr: Tensor,
      pageIndices: Tensor,
      qoIndptr: Tensor,
      output: Tensor,
      lse: Tensor,
      smScale: number,
      causal: number
    ): void => {
      // TODO: Implement actual WebInfer prefill kernel
    }
  );

  // ============================================================
  // Paged Decode Attention
  // ============================================================

  registerFunc(
    "webinfer_decode_plan",
    (
      batchSize: number,
      pageSize: number,
      numQoHeads: number,
      numKvHeads: number,
      qkHeadDim: number,
      vHeadDim: number
    ): string => {
      const planInfo: DecodePlanInfo = {
        key: `decode_${numQoHeads}_${numKvHeads}_${qkHeadDim}_${pageSize}`,
        batchSize,
        pageSize,
        numQoHeads,
        numKvHeads,
        qkHeadDim,
        vHeadDim,
      };
      return JSON.stringify(planInfo);
    }
  );

  registerFunc(
    "webinfer_decode_run",
    (
      planInfoJson: string,
      q: Tensor,
      pages: Tensor,
      pageIndptr: Tensor,
      pageIndices: Tensor,
      output: Tensor,
      lse: Tensor,
      smScale: number
    ): void => {
      // TODO: Implement actual WebInfer decode kernel
    }
  );

  // ============================================================
  // Ragged Prefill Attention
  // ============================================================

  registerFunc(
    "webinfer_ragged_prefill_plan",
    (
      batchSize: number,
      totalQoLen: number,
      numQoHeads: number,
      numKvHeads: number,
      qkHeadDim: number,
      vHeadDim: number,
      causal: boolean
    ): string => {
      const planInfo: RaggedPrefillPlanInfo = {
        key: `ragged_${numQoHeads}_${numKvHeads}_${qkHeadDim}`,
        batchSize,
        totalQoLen,
        numQoHeads,
        numKvHeads,
        qkHeadDim,
        vHeadDim,
        causal,
      };
      return JSON.stringify(planInfo);
    }
  );

  registerFunc(
    "webinfer_ragged_prefill_run",
    (
      planInfoJson: string,
      q: Tensor,
      k: Tensor,
      v: Tensor,
      qoIndptr: Tensor,
      kvIndptr: Tensor,
      output: Tensor,
      lse: Tensor,
      smScale: number,
      causal: number
    ): void => {
      // TODO: Implement actual WebInfer kernel execution
    }
  );
}

/**
 * Register GEMM kernels
 */
function registerGemmKernels(
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext
): void {
  registerFunc(
    "webinfer_gemm_run",
    (a: Tensor, b: Tensor, c: Tensor) => {
      if (!webinferCtx) {
        console.warn("WebInfer: context not initialized, skipping gemm");
        return;
      }

      const dtype = "float16";
      const webinferA = toWebinferTensor(a, webGPUContext, dtype);
      const webinferB = toWebinferTensor(b, webGPUContext, dtype);
      const webinferC = toWebinferTensor(c, webGPUContext, dtype);

      // Queue GPU commands (doesn't wait)
      webinfer.gemm.bmm_fp16(
        webinferCtx,
        webinferA,
        webinferB,
        webinferC
      );
    }
  );
}

/**
 * Register normalization kernels
 */
function registerNormKernels(
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext
): void {
  registerFunc(
    "webinfer_rmsnorm_run",
    (input: Tensor, weight: Tensor, output: Tensor, eps: number) => {
      if (!webinferCtx) {
        console.warn("WebInfer: context not initialized, skipping rmsnorm");
        return;
      }

      const dtype = "float16";
      const webinferInput = toWebinferTensor(input, webGPUContext, dtype);
      const webinferWeight = toWebinferTensor(weight, webGPUContext, dtype);
      const webinferOutput = toWebinferTensor(output, webGPUContext, dtype);

      // Queue GPU commands (doesn't wait)
      webinfer.norm.rmsnorm(
        webinferCtx,
        webinferInput,
        webinferWeight,
        webinferOutput,
        eps
      );
    }
  );
}

/**
 * Register activation kernels
 */
function registerActivationKernels(
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext
): void {
  registerFunc(
    "webinfer_silu_and_mul_run",
    (input: Tensor, output: Tensor) => {
      if (!webinferCtx) {
        console.warn("WebInfer: context not initialized, skipping silu_and_mul");
        return;
      }

      const dtype = "float16";
      const webinferInput = toWebinferTensor(input, webGPUContext, dtype);
      const webinferOutput = toWebinferTensor(output, webGPUContext, dtype);

      // Queue GPU commands (doesn't wait)
      webinfer.activation.silu_and_mul(
        webinferCtx,
        webinferInput,
        webinferOutput
      );
    }
  );
}

/**
 * Register RoPE kernels
 */
function registerRopeKernels(
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext
): void {
  registerFunc(
    "webinfer_rope_run",
    (
      q: Tensor,
      k: Tensor,
      positions: Tensor,
      qOut: Tensor,
      kOut: Tensor,
      ropeTheta: number,
      ropeScale: number
    ) => {
      if (!webinferCtx) {
        console.warn("WebInfer: context not initialized, skipping rope");
        return;
      }

      const dtype = "float16";
      const webinferQ = toWebinferTensor(q, webGPUContext, dtype);
      const webinferK = toWebinferTensor(k, webGPUContext, dtype);
      const webinferQOut = toWebinferTensor(qOut, webGPUContext, dtype);
      const webinferKOut = toWebinferTensor(kOut, webGPUContext, dtype);

      // Queue GPU commands (doesn't wait)
      webinfer.rope.apply_rope(
        webinferCtx,
        webinferQ,
        webinferK,
        positions,
        webinferQOut,
        webinferKOut,
        ropeTheta,
        ropeScale
      );
    }
  );
}

/**
 * Register a kernel from its specification
 */
async function registerKernelFromSpec(
  spec: KernelSpec,
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext
): Promise<void> {
  const key = getSpecKey(spec);

  // JIT compile the kernel using webinfer's JIT module
  const compiled = webinfer.jit.compileKernel(spec);

  // Store the compiled pipeline
  compiledPipelines.set(key, {
    spec,
    // Wrappers are created lazily in the run functions
  });
}

/**
 * Initialize WebInfer from kernel specs embedded in module.
 *
 * This function takes a JSON string of kernel specs (from module attribute)
 * and pre-compiles all the WGSL shaders, then registers the kernel functions.
 *
 * @param registerFunc Function to register a global PackedFunc
 * @param webGPUContext The WebGPU context for buffer access
 * @param specsJson JSON string of kernel specifications
 */
export async function initWebinferFromSpecs(
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext,
  specsJson: string
): Promise<void> {
  // Initialize webinfer context if not already done
  if (!webinferCtx) {
    try {
      webinferCtx = webinfer.WebInferContext.fromDevice(webGPUContext.device);
    } catch (error) {
      console.warn("WebInfer: failed to create context from device:", error);
      // Continue anyway - we still need to register the kernel functions
    }
  }

  // Parse specs - handle both array and single object
  let specs: KernelSpec[];
  const parsed = JSON.parse(specsJson);
  if (Array.isArray(parsed)) {
    specs = parsed;
  } else {
    specs = [parsed];
  }

  // Pre-compile all kernels from specs
  try {
    kernelRegistry = await webinfer.initFromSpecs(webGPUContext.device, specs);
  } catch (error) {
    console.warn("WebInfer: failed to compile kernels from specs:", error);
    // Continue anyway to register the kernel functions
  }

  // Register all kernel functions (they're needed even if pre-compilation failed)
  registerAttentionKernels(registerFunc, webGPUContext);
  registerGemmKernels(registerFunc, webGPUContext);
  registerNormKernels(registerFunc, webGPUContext);
  registerActivationKernels(registerFunc, webGPUContext);
  registerRopeKernels(registerFunc, webGPUContext);
}

/**
 * Get the kernel registry (for use by kernel implementations).
 */
export function getKernelRegistry(): webinfer.CompiledKernelRegistry | null {
  return kernelRegistry;
}

/**
 * Legacy initialization function for backward compatibility
 */
export async function initWebinferKernels(
  registerFunc: (name: string, func: (...args: any[]) => any) => void,
  webGPUContext: WebGPUContext
): Promise<void> {
  return initWebinferFromModule(registerFunc, webGPUContext);
}

/**
 * Check if webinfer is available and initialized
 */
export function isWebinferAvailable(): boolean {
  return webinferCtx !== null;
}

/**
 * Get the webinfer context (for direct access if needed)
 */
export function getWebinferContext(): webinfer.WebInferContext | null {
  return webinferCtx;
}
