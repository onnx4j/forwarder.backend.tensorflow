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
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.BatchNormalizationV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.nn.FusedBatchNorm;

public class TFBatchNormalizationV1 extends TFAiOnnxOperator<Tensor<Number>> implements BatchNormalizationV1 {

	@Override
	public OperatorOutputs<Tensor<Number>> forward(Node node, Inputs inputs) {
		BatchNormalizationInputsV1<Tensor<Number>> castedOperatorInputs = new BatchNormalizationInputsV1<Tensor<Number>>(node, inputs);
		Tensor<Number> x = castedOperatorInputs.getX();
		Tensor<Number> scale = castedOperatorInputs.getScale();
		Tensor<Number> b = castedOperatorInputs.getB();
		Tensor<Number> mean = castedOperatorInputs.getMean();
		Tensor<Number> var = castedOperatorInputs.getVar();
		Float epsilon = castedOperatorInputs.getEpsilon();
		Boolean isTest = castedOperatorInputs.isTest();
		Float momentum = castedOperatorInputs.getMomentum();
		Boolean spatial = castedOperatorInputs.isSpatial();
		List<Long> consumedInputs = castedOperatorInputs.getConsumedInputs();
		return new BatchNormalizationOutputsV1<Tensor<Number>>(
				this.batchNormalization(x, scale, b, mean, var, consumedInputs, epsilon, isTest, momentum, spatial));
	}

	public Tensor<Number> batchNormalization(Tensor<Number> x, Tensor<Number> scale, Tensor<Number> b, Tensor<Number> mean, Tensor<Number> var,
			List<Long> consumedInputs, Float epsilon, Boolean isTest, Float momentum, Boolean spatial) {
		TFOps tfOps = TFSession.getOps();
		Operand<Number> opX = tfOps.constant(x);
		Operand<Number> opMean = tfOps.constant(mean);
		Operand<Number> opVar = tfOps.constant(var);
		Operand<Number> opBeta = tfOps.constant(b);
		Operand<Number> opGamma = tfOps.constant(scale);
		Operand<Number> opFusedBatchNorm = tfOps.ops().nn.fusedBatchNorm(
				tfOps.toNHWC(opX), 
				opGamma, 
				opBeta, 
				opMean, 
				opVar, 
				FusedBatchNorm
					.dataFormat("NHWC")
					.epsilon(epsilon)
					.isTraining(false)
			).y();
		/*Operation opBatchNorm = scope.env()
			.opBuilder("tf.nn.batch_normalization", scope.makeOpName("BatchNormWithGlobalNormalization"))
			.addInput(TensorUtil.toConstant(scope, x).asOutput())
			.addInput(TensorUtil.toConstant(scope, mean).asOutput())
			.addInput(TensorUtil.toConstant(scope, var).asOutput())
			.addInput(TensorUtil.toConstant(scope, b).asOutput())
			.addInput(TensorUtil.toConstant(scope, scale).asOutput())
			.setAttr("variance_epsilon", epsilon)
		    .setAttr("scale_after_normalization", true)
			.build();*/
		//Operand opBatchNorm = BatchNormWithGlobalNormalization.create(scope, opX, opMean, opVar, opBeta, opGamma,
		//		epsilon, true);
		return opFusedBatchNorm.asOutput().tensor();
	}

}