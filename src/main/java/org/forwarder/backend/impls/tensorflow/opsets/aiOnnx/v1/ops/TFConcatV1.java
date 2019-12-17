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

import java.util.LinkedList;
import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.aiOnnx.v1.ops.ConcatV1;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Concat;
import org.tensorflow.op.core.Constant;

public class TFConcatV1 extends TFOperator implements ConcatV1<Tensor<?>> {

	@Override
	public Tensor<?> concat(List<Tensor<?>> inputs, Long axis) {
		Scope scope = new Scope(TFSession.get());
		List constants = new LinkedList();
		for (Tensor<?> tensor : inputs) {
			constants.add(TensorUtil.toConstant(scope, tensor));
		}

		return Concat.create(scope, constants, Constant.create(scope, axis)).asOutput().tensor();
	}

}