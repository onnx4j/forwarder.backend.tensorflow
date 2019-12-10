package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.onnx4j.opsets.v1.ops.ConstantV1;
import org.tensorflow.Tensor;

public class TFConstantV1 extends TFOperator implements ConstantV1<Tensor<?>> {

	@Override
	public Tensor<?> constant(org.onnx4j.Tensor x0) {
		return TFSession.getSession().getBackend().toBackendTensor(x0);
	}

}
