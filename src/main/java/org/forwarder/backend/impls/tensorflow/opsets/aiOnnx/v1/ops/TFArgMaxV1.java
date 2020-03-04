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
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ArgMaxV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.math.ArgMax;

public class TFArgMaxV1 extends TFAiOnnxOperator<Tensor<Long>> implements ArgMaxV1 {

	@Override
	public OperatorOutputs<Tensor<Long>> forward(Node node, Inputs inputs) {
		ArgMaxInputsV1<Tensor<Number>> castedOperatorInputs = new ArgMaxInputsV1<Tensor<Number>>(node, inputs);
		Tensor<Number> data = castedOperatorInputs.getData();
		Long axis = castedOperatorInputs.getAxis();
		Long keepdims = castedOperatorInputs.getKeepdims();
		return new ArgMaxOutputV1<Tensor<Long>>(this.argmax(data, axis, keepdims));
	}

	protected Tensor<Long> argmax(Tensor<Number> x0, Long axis, Long keepdims) {
		TFOps tfOps = TFSession.getOps();
		Operand<Number> operandX0 = tfOps.constant(x0);
		ArgMax<Long> operandArgMax = tfOps.ops().math.argMax(operandX0, tfOps.ops().constant(keepdims));
		return operandArgMax.asOutput().tensor();
	}

}