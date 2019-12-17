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

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.aiOnnx.v1.ops.ImageScalerV1;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.math.Mul;

@Deprecated
public class TFImageScalerV1 extends TFOperator implements ImageScalerV1<Tensor<?>> {

	@Override
	public Tensor<?> scale(Tensor<?> input, Float scale, List<Float> bias) {
		Scope scope = new Scope(TFSession.get());
		
		Operand x = TensorUtil.toConstant(scope, input);
		Operand y = Constant.create(scope, scale);
		//BiasAdd.create(scope, value, bias, BiasAdd.dataFormat(dataFormat))
		return Mul.create(scope, x, y).asOutput().tensor();
	}

}