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

/*!
 * \file webinfer_runtime.cc
 * \brief WebInfer runtime module for WASM/Emscripten builds.
 *
 * This file provides the module loader for WebInfer source modules
 * when running in a WASM environment. It reads the kernel specs from
 * the serialized bytes and creates a module that can be accessed by
 * the JavaScript runtime.
 */

#include <dmlc/memory_io.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace runtime {

/*!
 * \brief WebInfer module node for WASM runtime.
 *
 * This module stores kernel specifications as JSON that will be
 * JIT compiled to WGSL shaders by the JavaScript WebInfer library.
 */
class WebInferModuleNode final : public ffi::ModuleObj {
 public:
  /*!
   * \brief Constructor
   * \param kernel_specs JSON string containing kernel specifications
   */
  explicit WebInferModuleNode(std::string kernel_specs) : kernel_specs_(std::move(kernel_specs)) {}

  /*!
   * \brief Get the type key for this module
   * \return "webinfer"
   */
  const char* kind() const final { return "webinfer"; }

  /*!
   * \brief Get the property mask of this module
   * \return Property mask indicating binary serializability
   */
  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  /*!
   * \brief Get a function from this module
   *
   * This module provides:
   * - "get_kernel_specs": Returns the kernel specifications JSON
   *
   * \param name Function name to look up
   * \return The function if found, None otherwise
   */
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    if (name == "get_kernel_specs") {
      std::string specs = kernel_specs_;
      return ffi::Function(
          [specs](ffi::PackedArgs args, ffi::Any* rv) { *rv = ffi::String(specs); });
    }
    return ffi::Function(nullptr);
  }

  /*!
   * \brief Serialize this module to bytes
   * \return Serialized bytes
   */
  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    dmlc::MemoryStringStream ms(&buffer);
    dmlc::Stream* stream = &ms;
    stream->Write(kernel_specs_);
    return ffi::Bytes(buffer);
  }

  /*!
   * \brief Get source code for inspection
   * \param format The format to inspect ("json" returns kernel specs)
   * \return The source string
   */
  ffi::String InspectSource(const ffi::String& format) const final {
    // Return the kernel specs JSON for any format
    return kernel_specs_;
  }

 private:
  /*! \brief JSON string containing kernel specifications */
  std::string kernel_specs_;
};

// Global storage for all loaded kernel specs (as JSON array)
// This allows JS runtime to retrieve specs via a global function
static std::vector<std::string> g_webinfer_kernel_specs_list;

/*!
 * \brief Get all kernel specs from loaded webinfer modules
 * \return Kernel specs as JSON array string
 */
ffi::String WebInferGetKernelSpecs() {
  // Combine all specs into a JSON array
  std::string result = "[";
  for (size_t i = 0; i < g_webinfer_kernel_specs_list.size(); ++i) {
    if (i > 0) result += ",";
    result += g_webinfer_kernel_specs_list[i];
  }
  result += "]";
  return ffi::String(result);
}

/*!
 * \brief Clear all stored kernel specs (for testing/reset)
 */
void WebInferClearKernelSpecs() {
  g_webinfer_kernel_specs_list.clear();
}

/*!
 * \brief Load WebInfer module from bytes
 *
 * This function is called by the TVM module loader when loading
 * a module of kind "webinfer". It deserializes the kernel specs
 * and creates a WebInferModuleNode.
 *
 * It also stores the specs globally so they can be retrieved via
 * the "webinfer.get_kernel_specs" global function.
 *
 * \param bytes Serialized module bytes
 * \return Loaded module
 */
ffi::Module WebInferModuleLoadFromBytes(const ffi::Bytes& bytes) {
  dmlc::MemoryFixedSizeStream ms(const_cast<char*>(bytes.data()), bytes.size());
  dmlc::Stream* stream = &ms;
  std::string kernel_specs;
  stream->Read(&kernel_specs);

  // Add specs to global list for easy retrieval by JS runtime
  g_webinfer_kernel_specs_list.push_back(kernel_specs);

  return ffi::Module(ffi::make_object<WebInferModuleNode>(kernel_specs));
}

// Register the module loader and kernel specs getter
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.Module.load_from_bytes.webinfer", WebInferModuleLoadFromBytes);
  refl::GlobalDef().def("webinfer.get_kernel_specs", WebInferGetKernelSpecs);
}

}  // namespace runtime
}  // namespace tvm
