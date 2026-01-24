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
 * \file webinfer_source_module.cc
 * \brief WebInfer source module for TVM.
 *
 * This module stores kernel specifications as JSON that will be
 * JIT compiled to WGSL shaders by the JavaScript WebInfer library.
 *
 * Architecture:
 * - Python generates kernel specs (not compiled code)
 * - Specs are serialized as JSON and stored in this module
 * - JavaScript runtime loads specs and JIT compiles WGSL
 * - TVM runtime resolves ExternFunc to WebInfer functions
 *
 * This is similar to how FlashInfer integrates with TVM:
 * - FlashInfer: Python -> CUDA JIT -> .o files -> load_static_library()
 * - WebInfer: Python -> kernel specs (JSON) -> WebInferSourceModule -> JS JIT -> WGSL
 */

#include <dmlc/memory_io.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief WebInfer source module that stores kernel specifications.
 *
 * The kernel specifications are JSON strings that describe kernels
 * to be JIT compiled by the JavaScript WebInfer library. This module
 * is serialized into the TVM module and loaded by the JS runtime.
 */
class WebInferSourceModuleNode final : public ffi::ModuleObj {
 public:
  /*!
   * \brief Constructor
   * \param kernel_specs JSON string containing kernel specifications
   */
  explicit WebInferSourceModuleNode(ffi::String kernel_specs) : kernel_specs_(kernel_specs) {}

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
   * This module provides "get_kernel_specs" function to retrieve
   * the kernel specifications JSON.
   *
   * \param name Function name to look up
   * \return The function if found, None otherwise
   */
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    if (name == "get_kernel_specs") {
      ffi::String specs = kernel_specs_;
      return ffi::Function(
          [specs](ffi::PackedArgs args, ffi::Any* rv) { *rv = specs; });
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
    stream->Write(std::string(kernel_specs_));
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

  /*!
   * \brief Load module from binary stream
   * \param strm Pointer to the stream
   * \return Loaded module
   */
  static ffi::Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string kernel_specs;
    ICHECK(stream->Read(&kernel_specs)) << "WebInferSourceModule: Failed to read kernel specs";
    return ffi::Module(ffi::make_object<WebInferSourceModuleNode>(kernel_specs));
  }

 private:
  /*! \brief JSON string containing kernel specifications */
  ffi::String kernel_specs_;
};

// Register the module creation function
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  // Register function to create WebInferSourceModule from kernel specs
  refl::GlobalDef().def("runtime.WebInferSourceModuleCreate", [](ffi::String kernel_specs) {
    return ffi::Module(ffi::make_object<WebInferSourceModuleNode>(kernel_specs));
  });

  // Register function to load WebInferSourceModule from binary
  refl::GlobalDef().def("runtime.module.loadbinary_webinfer",
                        WebInferSourceModuleNode::LoadFromBinary);
}

}  // namespace runtime
}  // namespace tvm
