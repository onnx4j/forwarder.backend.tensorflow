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

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.forwarder.backend.impls.tensorflow.TFOps;
import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.TFAiOnnxOperator;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AveragePoolV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MaxPoolV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.nn.AvgPool;

import com.google.common.collect.Lists;

public class TFAveragePoolV1 extends TFAiOnnxOperator<Tensor<Number>> implements AveragePoolV1 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		AveragePoolInputsV1<Tensor<Number>> castedOperatorInputs = new AveragePoolInputsV1<Tensor<Number>>(node,
				inputs);
		Tensor<Number> x = castedOperatorInputs.getX();
		String autoPad = castedOperatorInputs.getAutoPad();
		List<Long> kernelShape = castedOperatorInputs.getKernelShape();
		List<Long> pads = castedOperatorInputs.getPads();
		List<Long> strides = castedOperatorInputs.getStrides();
		return new AveragePoolOutputV1<Tensor<Number>>(this.averagePool(x, autoPad, kernelShape, pads, strides));
	}

	protected Tensor<Number> averagePool(Tensor<Number> data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides) {
		TFOps tfOps = TFSession.getOps();
		Operand<Number> constantData = tfOps.constant(data);

		//
		// Add 1L to first and last of kernelShape=[m, n]
		// =>
		// [1, m, n, 1]
		//
		LinkedList<Long> newKernelShape = Lists.newLinkedList(kernelShape);
		newKernelShape.addFirst(1L);
		newKernelShape.addLast(1L);

		//
		// Add 1L to first and last of strides=[m, n]
		// =>
		// [1, m, n, 1]
		//
		LinkedList<Long> newStrides = Lists.newLinkedList(strides);
		newStrides.addFirst(1L);
		newStrides.addLast(1L);

		Operand<Number> opAvgPool = tfOps.ops().nn.avgPool(tfOps.toNHWC(constantData), newKernelShape, newStrides,
				this.getTFPadding(data, autoPad, kernelShape, pads, strides), AvgPool.dataFormat("NHWC"));
		return tfOps.toNCHW(opAvgPool).asOutput().tensor();
	}

	private String getTFPadding(Tensor<Number> data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides) {
		if ("NOTSET".equalsIgnoreCase(autoPad)) {
			if (pads != null && pads.size() > 0) {
				int nbSpatialSize = kernelShape.size();
				Long[] validPads = new Long[nbSpatialSize * 2];
				Arrays.fill(validPads, 0, validPads.length, 0L);

				if (Arrays.deepEquals(validPads, pads.toArray(new Long[pads.size()])))
					return "VALID";

				throw new NotImplementedException(String.format("[%s] Tensorflow can not support \"%s\" padding mode",
						AveragePoolV1.OP_TYPE, autoPad));
			} else {
				return "VALID";
			}
		} else {
			if ("SAME_UPPER".equalsIgnoreCase(autoPad) || "SAME_LOWER".equalsIgnoreCase(autoPad))
				return "SAME";
			else if ("VALID".equalsIgnoreCase(autoPad))
				return "VALID";
			else
				throw new NotImplementedException(String.format("[%s] Tensorflow can not support \"%s\" padding mode",
						MaxPoolV1.OP_TYPE, autoPad));
		}
	}

}