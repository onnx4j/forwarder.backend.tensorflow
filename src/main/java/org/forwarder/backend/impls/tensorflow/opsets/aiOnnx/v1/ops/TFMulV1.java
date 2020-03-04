/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for mulitional information regarding copyright ownership.
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
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MulV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;

public class TFMulV1 extends TFOperator<Tensor<Number>> implements MulV1 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		MulInputsV1<Tensor<Number>> castedOperatorInputs = new MulInputsV1<Tensor<Number>>(node, inputs);
		Tensor<Number> a = castedOperatorInputs.getA();
		Tensor<Number> b = castedOperatorInputs.getB();
		Long axis = castedOperatorInputs.getAxis();
		Long broadcast = castedOperatorInputs.getBroadcast();
		//List<Long> consumedInputs = castedOperatorInputs.getConsumedInputs();
		return new MulOutputV1<Tensor<Number>>(this.mul(a, b, axis, broadcast));
	}

	protected Tensor<Number> mul(Tensor<Number> a, Tensor<Number> b, Long axis, Long broadcast) {
		TFOps tfOps = TFSession.getOps();

		Operand<Number> operandA = tfOps.constant(a);
		Operand<Number> operandB;
		if (broadcast == 1L) {
			operandB = tfOps.broadcast(tfOps, a, b, axis);
		} else {
			operandB = tfOps.constant(b);
		}

		Operand<Number> operandMul = tfOps.ops().math.mul(operandA, operandB);
		return operandMul.asOutput().tensor();
	}

}