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
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AbsV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Tensor;
import org.tensorflow.op.math.Abs;

public class TFAbsV1 extends TFAiOnnxOperator<Tensor<Number>> implements AbsV1 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		AbsInputsV1<Tensor<Number>> castedOperatorInputs = new AbsInputsV1<Tensor<Number>>(node, inputs);
		Tensor<Number> x = castedOperatorInputs.getX();
		return new AbsOutputV1<Tensor<Number>>(this.abs(x));
	}

	protected Tensor<Number> abs(Tensor<Number> x) {
		TFOps tfOps = TFSession.getOps();
		Abs<Number> abs = tfOps.ops().math.abs(tfOps.constant(x));
		return abs.asOutput().tensor();
	}

}