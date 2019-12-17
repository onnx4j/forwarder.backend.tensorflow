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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v7.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFAveragePoolV1;
import org.onnx4j.opsets.aiOnnx.v7.ops.AveragePoolV7;
import org.tensorflow.Tensor;

public class TFAveragePoolV7 extends TFAveragePoolV1 implements AveragePoolV7<Tensor<?>> {

	@Override
	public Tensor<?> averagePool(Tensor<?> data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides, Long countIncludePad) {
		if (countIncludePad != null && countIncludePad != 0L)
			throw new UnsupportedOperationException(
					String.format("[%s] Unable to handle \"countIncludePad\" is not equals to 0L", OP_TYPE));

		return super.averagePool(data, autoPad, kernelShape, pads, strides);
	}

}