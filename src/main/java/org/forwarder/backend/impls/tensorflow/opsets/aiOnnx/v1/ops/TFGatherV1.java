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

import org.forwarder.backend.impls.tensorflow.TFOps;
import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.TFAiOnnxOperator;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.GatherV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class TFGatherV1 extends TFAiOnnxOperator<Tensor<Number>> implements GatherV1 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		GatherInputsV1<Tensor<Number>> castedOperatorInputs = new GatherInputsV1<Tensor<Number>>(node, inputs);
		Tensor<Number> data = castedOperatorInputs.getData();
		Tensor<Number> indices = castedOperatorInputs.getIndices();
		Long axis = castedOperatorInputs.getAxis();
		return new GatherOutputV1<Tensor<Number>>(this.gather(data, indices, axis));
	}

	protected Tensor<Number> gather(Tensor<Number> data, Tensor<Number> indices, Long axis) {
		TFOps tfOps = TFSession.getOps();
		Operand<Number> opParams = tfOps.constant(data);
		Operand<Number> opIndices = tfOps.constant(indices);
		Operand<Long> opAxis = tfOps.constant(Tensors.create(axis));
		return tfOps.ops().gather(opParams, opIndices, opAxis).asOutput().tensor();
	}

}