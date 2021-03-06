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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v4.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFConcatV1;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v4.ops.ConcatV4;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Tensor;

public class TFConcatV4 extends TFConcatV1 implements ConcatV4 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		ConcatInputsV4<Tensor<Number>> castedOperatorInputs = new ConcatInputsV4<Tensor<Number>>(node, inputs);
		List<Tensor<Number>> inputList = castedOperatorInputs.getInputs();
		Long axis = castedOperatorInputs.getAxis();
		return new ConcatOutputV4<Tensor<Number>>(super.concat(inputList, axis));
	}

}