/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v6.ops;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFMulV1;
import org.onnx4j.opsets.aiOnnx.v6.ops.MulV6;
import org.tensorflow.Tensor;

public class TFMulV6 extends TFMulV1 implements MulV6<Tensor<?>> {

	@Override
	public Tensor<?> mul(Tensor<?> a, Tensor<?> b, Long axis, Long broadcast) {
		return super.mul(a, b, axis, broadcast, null);
	}

}