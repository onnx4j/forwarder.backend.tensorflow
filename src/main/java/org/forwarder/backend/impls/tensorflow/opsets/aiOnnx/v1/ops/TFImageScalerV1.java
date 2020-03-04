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
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ImageScalerV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.nn.BiasAdd;

import com.google.common.primitives.Floats;

@Deprecated
public class TFImageScalerV1 extends TFAiOnnxOperator<Tensor<Float>> implements ImageScalerV1 {

	@Override
	public OperatorOutputs<Tensor<Float>> forward(Node node, Inputs inputs) {
		ImageScalerInputsV1<Tensor<Float>> castedOperatorInputs = new ImageScalerInputsV1<Tensor<Float>>(node, inputs);
		Tensor<Float> input = castedOperatorInputs.getInput();
		Float scale = castedOperatorInputs.getScale();
		List<Float> bias = castedOperatorInputs.getBias();
		return new ImageScalerOutputV1<Tensor<Float>>(this.scale(input, scale, bias));
	}

	protected Tensor<Float> scale(Tensor<Float> input, Float scale, List<Float> bias) {
		TFOps tfOps = TFSession.getOps();

		Operand<Float> opX = tfOps.constant(input);
		Operand<Float> opY = tfOps.ops().constant(scale);
		Operand<Float> opMul = tfOps.ops().math.mul(opX, opY);
		if (bias != null && bias.size() > 0) {
			return tfOps.ops().nn.biasAdd(opMul, tfOps.ops().constant(Floats.toArray(bias)), BiasAdd.dataFormat("NCHW"))
					.asOutput().tensor();
		} else {
			return opMul.asOutput().tensor();
		}
	}

}