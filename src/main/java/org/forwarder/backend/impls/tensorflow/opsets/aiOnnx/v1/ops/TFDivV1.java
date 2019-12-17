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

import org.apache.commons.lang3.Range;
import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.aiOnnx.v1.ops.DivV1;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.ExpandDims;
import org.tensorflow.op.math.Div;

public class TFDivV1 extends TFOperator implements DivV1<Tensor<?>> {

	@Override
	public Tensor<?> div(Tensor<?> a, Tensor<?> b, Long axis, Long broadcast, List<Long> consumedInputs) {
		Scope scope = new Scope(TFSession.get());

		Operand<Number> opTensorB;
		if (broadcast == 1L) {
			opTensorB = TensorUtil.toConstant(scope,
					this.broadcast(scope, (Tensor<Number>) a, (Tensor<Number>) b, axis));
		} else {
			opTensorB = TensorUtil.toConstant(scope, (Tensor<Number>) b);
		}

		return Div.create(scope, TensorUtil.toConstant(scope, (Tensor<Number>) a), opTensorB).asOutput().tensor();
	}

	private <T> Tensor<T> broadcast(Scope scope, Tensor<T> a, Tensor<T> b, Long axis) {
		if (axis == null)
			return b;

		if (axis + b.numDimensions() == a.numDimensions())
			return b;

		if (axis < 0L)
			axis += a.numDimensions();

		Range<Integer> keepdims = Range.between(axis.intValue(), axis.intValue() + b.numDimensions() - 1);
		Operand<T> opTensorB = TensorUtil.toConstant(scope, b);
		int nbDiffDims = a.numDimensions() - b.numDimensions();
		for (int n = 0; n <= nbDiffDims; n++) {
			if (keepdims.contains(n) == false) {
				Operand<Long> opAxis = Constant.create(scope, Long.valueOf(n));
				opTensorB = ExpandDims.create(scope, opTensorB, opAxis).asOutput();
			}
		}
		return opTensorB.asOutput().tensor();
	}

}