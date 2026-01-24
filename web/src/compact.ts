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
/** NodeJS and Web compact layer */
import { LibraryProvider } from "./types";
import EmccWASI from "./tvmjs_runtime_wasi";

// Dynamic require that bundlers won't analyze/hoist
// This prevents rollup from converting conditional requires to static imports
const dynamicRequire = typeof require !== "undefined" ? require : null;

/**
 * Get performance measurement.
 */
export function getPerformance(): Performance {
  if (typeof performance === "undefined") {
    // Node.js environment - use perf_hooks
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const performanceNode = dynamicRequire!("perf_hooks");
    return performanceNode.performance as Performance;
  } else {
    return performance as Performance;
  }
}

/**
 * Create a new websocket for a given URL
 * @param url The url.
 */
export function createWebSocket(url: string): WebSocket {
  if (typeof WebSocket === "undefined") {
    // Node.js environment - use ws package
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const WS = dynamicRequire!("ws");
    return new WS(url);
  } else {
    return new (WebSocket as any)(url);
  }
}

/**
 * Create a WASI based on current environment.
 *
 * @return A wasi that can run on broswer or local.
 */
export function createPolyfillWASI(): LibraryProvider {
  return new EmccWASI();
}
