package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.AbsV1;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.math.Abs;

public class TFAbsV1 extends TFOperator implements AbsV1<Tensor<?>> {

	@Override
	public Tensor<?> abs(Tensor<?> x) {
		Scope scope = new Scope(TFSession.get());
		return Abs.create(scope, TensorUtil.toConstant(scope, (Tensor<Number>) x)).asOutput().tensor();
	}

}
