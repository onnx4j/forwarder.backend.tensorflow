package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.ArgMaxV1;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.math.ArgMax;

public class TFArgMaxV1 extends TFOperator implements ArgMaxV1<Tensor<?>> {

	@Override
	public Tensor<?> argmax(Tensor<?> x, int axis, int keepdims) {
		Scope scope = new Scope(TFSession.get());
		return ArgMax.create(scope, TensorUtil.toConstant(scope, x), Constant.create(scope, axis)).asOutput().tensor();
	}

}
