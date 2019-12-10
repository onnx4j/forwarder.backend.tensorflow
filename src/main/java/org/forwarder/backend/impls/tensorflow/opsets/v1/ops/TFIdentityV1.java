package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.IdentityV1;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Identity;

public class TFIdentityV1 extends TFOperator implements IdentityV1<Tensor<?>> {

	@Override
	public Tensor<?> identity(Tensor<?> x0) {
		Scope scope = new Scope(TFSession.get());
		return Identity.create(scope, TensorUtil.toConstant(scope, x0)).asOutput().tensor();
	}

}
