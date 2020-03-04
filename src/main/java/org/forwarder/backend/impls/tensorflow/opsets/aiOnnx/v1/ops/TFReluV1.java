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

import org.forwarder.backend.impls.tensorflow.TFOps;
import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.TFAiOnnxOperator;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReluV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Tensor;

public class TFReluV1 extends TFAiOnnxOperator<Tensor<Number>> implements ReluV1 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		ReluInputsV1<Tensor<Number>> castedOperatorInputs = new ReluInputsV1<Tensor<Number>>(node, inputs);
		Tensor<Number> x = castedOperatorInputs.getX();
		List<Long> consumedInputs = castedOperatorInputs.getConsumedInputs();
		return new ReluOutputV1<Tensor<Number>>(this.relu(x, consumedInputs));
	}

	protected Tensor<Number> relu(Tensor<Number> x, List<Long> consumed_inputs) {
		TFOps tfOps = TFSession.getOps();
		return tfOps.ops().nn.relu(tfOps.constant(x)).asOutput().tensor();
	}

}