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
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ShapeV1;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.tensorflow.Tensor;

public class TFShapeV1 extends TFOperator<Tensor<Long>> implements ShapeV1 {

	@Override
	public OperatorOutputs<Tensor<Long>> forward(Node node, Inputs inputs) {
		ShapeInputsV1<Tensor<Long>> castedOperatorInputs = new ShapeInputsV1<Tensor<Long>>(node, inputs);
		Tensor<Long> data = castedOperatorInputs.getData();
		return new ShapeOutputV1<Tensor<Long>>(this.shape(data));
	}
	
	public Tensor<Long> shape(Tensor<?> data) {
		TFOps tfOps = TFSession.getOps();
		return tfOps.ops().shape(tfOps.constant(data), Long.class).asOutput().tensor();
	}

}