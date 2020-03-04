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

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.TFAiOnnxOperator;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConstantV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Tensor;

public class TFConstantV1 extends TFAiOnnxOperator<Tensor<?>> implements ConstantV1 {

	@Override
	public OperatorOutputs<Tensor<?>> forward(Node node, Inputs inputs) {
		ConstantInputsV1<Tensor<Object>> castedOperatorInputs = new ConstantInputsV1<Tensor<Object>>(node, inputs);
		org.onnx4j.Tensor value = castedOperatorInputs.getValue();
		return new ConstantOutputV1<Tensor<?>>(this.constant(value));
	}

	protected Tensor<?> constant(org.onnx4j.Tensor x0) {
		TFSession session = TFSession.getSession();
		return session.getBackend().toBackendTensor(session.getTensorManager(), x0);
	}

}