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

import java.util.LinkedList;
import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFOps;
import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.TFAiOnnxOperator;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConvV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.nn.BiasAdd;
import org.tensorflow.op.nn.Conv2d;

import com.google.common.collect.Lists;

public class TFConvV1 extends TFAiOnnxOperator<Tensor<Number>> implements ConvV1 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		ConvInputsV1<Tensor<Number>> castedOperatorInputs = new ConvInputsV1<Tensor<Number>>(node, inputs);
		Tensor<Number> x = castedOperatorInputs.getX();
		Tensor<Number> b = castedOperatorInputs.getB();
		Tensor<Number> w = castedOperatorInputs.getW();
		String autoPad = castedOperatorInputs.getAutoPad();
		List<Long> dilations = castedOperatorInputs.getDilations();
		Long group = castedOperatorInputs.getGroup();
		List<Long> kernelShape = castedOperatorInputs.getKernelShape();
		List<Long> pads = castedOperatorInputs.getPads();
		List<Long> strides = castedOperatorInputs.getStrides();
		return new ConvOutputV1<Tensor<Number>>(this.conv(x, w, b, autoPad, dilations, group, kernelShape, pads, strides));
	}

	protected Tensor<Number> conv(Tensor<Number> x, Tensor<Number> w, Tensor<Number> b, String autoPad, List<Long> dilations, Long group,
			List<Long> kernelShape, List<Long> pads, List<Long> strides) {
		TFOps tfOps = TFSession.getOps();
		
		//
		// Translate inputs from (N x C x KH x KW) to (N X KH x KW X C)
		// Tensorflow for java does not support NCHW mode on CPU temporarily
		//
		Operand<Number> operandX = tfOps.toNHWC(tfOps.constant(x));
		
		//
		// Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
		//
		Operand<Number> operandW = tfOps.toHWCN(tfOps.constant(w));

		//
		// Add 1L to first and last of dilations=[m, n]
		// =>
		// [1, m, n, 1]
		//
		LinkedList<Long> newDilations = Lists.newLinkedList(dilations);
		newDilations.addFirst(1L);
		newDilations.addLast(1L);

		//
		// Add 1L to first and last of strides=[m, n]
		// =>
		// [1, m, n, 1]
		//
		LinkedList<Long> newStrides = Lists.newLinkedList(strides);
		newStrides.addFirst(1L);
		newStrides.addLast(1L);

		Conv2d.Options options = Conv2d
				.useCudnnOnGpu(false)
				.dataFormat("NHWC")
				.dilations(newDilations);
		Operand<Number> opreandConv2D = tfOps.ops().nn.conv2d(
				operandX, 
				operandW, 
				newStrides,
				this.getTFPadding(x, w, b, autoPad, dilations, group, kernelShape, pads, strides), 
				options);
		
		if (b == null) {
			return tfOps.toNCHW(opreandConv2D).asOutput().tensor();
		} else {
			Operand<Number> opB = tfOps.constant(b);
			return tfOps.toNCHW(tfOps.ops().nn.biasAdd(opreandConv2D, opB, BiasAdd.dataFormat("NHWC"))).asOutput().tensor();
		}
	}

	protected String getTFPadding(Tensor<Number> x, Tensor<Number> w, Tensor<Number> b, String autoPad, List<Long> dilations, Long group,
			List<Long> kernelShape, List<Long> pads, List<Long> strides) {
		if ("VALID".equalsIgnoreCase(autoPad))
			return "VALID";
		else if ("SAME_UPPER".equalsIgnoreCase(autoPad) || "SAME_LOWER".equalsIgnoreCase(autoPad))
			return "SAME";
		else
			throw new RuntimeException(String.format("Tensorflow can not support \"%s\" padding mode", autoPad));
	}

}