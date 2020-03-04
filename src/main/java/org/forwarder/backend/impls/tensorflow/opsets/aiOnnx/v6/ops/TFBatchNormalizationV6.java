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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v6.ops;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFBatchNormalizationV1;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v6.ops.BatchNormalizationV6;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Tensor;

public class TFBatchNormalizationV6 extends TFBatchNormalizationV1 implements BatchNormalizationV6 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		BatchNormalizationInputsV6<Tensor<Number>> castedOperatorInputs = new BatchNormalizationInputsV6<Tensor<Number>>(
				node, inputs);
		Tensor<Number> x = castedOperatorInputs.getX();
		Tensor<Number> scale = castedOperatorInputs.getScale();
		Tensor<Number> b = castedOperatorInputs.getB();
		Tensor<Number> mean = castedOperatorInputs.getMean();
		Tensor<Number> var = castedOperatorInputs.getVar();
		Float epsilon = castedOperatorInputs.getEpsilon();
		Float momentum = castedOperatorInputs.getMomentum();
		Boolean spatial = castedOperatorInputs.isSpatial();
		return new BatchNormalizationOutputsV6<Tensor<Number>>(
				this.batchNormalization(x, scale, b, mean, var, null, epsilon, true, momentum, spatial));
	}

}