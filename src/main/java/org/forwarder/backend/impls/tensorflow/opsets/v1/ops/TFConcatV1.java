package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.LinkedList;
import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.ConcatV1;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Concat;
import org.tensorflow.op.core.Constant;

public class TFConcatV1 extends TFOperator implements ConcatV1<Tensor<?>> {

	@Override
	public Tensor<?> concat(List<Tensor<?>> inputs, Long axis) {
		Scope scope = new Scope(TFSession.get());
		List constants = new LinkedList();
		for (Tensor<?> tensor : inputs) {
			constants.add(TensorUtil.toConstant(scope, tensor));
		}

		return Concat.create(scope, constants, Constant.create(scope, axis)).asOutput().tensor();
	}

}
