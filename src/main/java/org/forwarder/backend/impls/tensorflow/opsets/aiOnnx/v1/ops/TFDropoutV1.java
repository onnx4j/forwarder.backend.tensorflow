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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops;

import java.util.List;
import java.util.Optional;

import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.onnx4j.opsets.aiOnnx.v1.ops.DropoutV1;
import org.tensorflow.Tensor;

public class TFDropoutV1 extends TFOperator implements DropoutV1<Tensor<?>> {

	@Override
	public List<Tensor<?>> dropout(Tensor<?> data, Boolean isTest, Float ratio, List<Long> consumedInputs) {
		if (isTest == false)
			throw new UnsupportedOperationException("Can not run " + DropoutV1.OP_TYPE + " in not test mode");
		
		return this.wrapMultiOutputs(Optional.of(data), Optional.empty());
	}

}