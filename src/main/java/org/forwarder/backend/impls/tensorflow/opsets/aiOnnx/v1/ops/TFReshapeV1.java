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
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReshapeV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Tensor;

import com.google.common.primitives.Longs;

public class TFReshapeV1 extends TFOperator<Tensor<? extends Number>> implements ReshapeV1 {

	@Override
	public OperatorOutputs<Tensor<? extends Number>> forward(Node node, Inputs inputs) {
		ReshapeInputsV1<Tensor<? extends Number>> castedOperatorInputs = new ReshapeInputsV1<Tensor<? extends Number>>(node, inputs);
		Tensor<? extends Number> data = castedOperatorInputs.getData();
		List<Long> shape = castedOperatorInputs.getShape();
		List<Long> consumedInputs = castedOperatorInputs.getConsumedInputs();
		return new ReshapeOutputV1<Tensor<? extends Number>>(this.reshape(data, shape, consumedInputs));
	}

	protected Tensor<? extends Number> reshape(Tensor<? extends Number> a, List<Long> shape, List<Long> consumedInputs) {
		TFOps tfOps = TFSession.getOps();
		return tfOps.ops().reshape(tfOps.constant(a), tfOps.ops().constant(Longs.toArray(shape))).asOutput().tensor();
	}

}