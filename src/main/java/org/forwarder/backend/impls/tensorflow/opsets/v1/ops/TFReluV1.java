package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.ReluV1;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.nn.Relu;

public class TFReluV1 extends TFOperator implements ReluV1<Tensor<?>> {

	@Override
	public Tensor<?> relu(Tensor<?> x, List<Long> consumed_inputs) {
		Scope scope = new Scope(TFSession.get());
		return Relu.create(scope, TensorUtil.toConstant(scope, x)).asOutput().tensor();
	}

}
