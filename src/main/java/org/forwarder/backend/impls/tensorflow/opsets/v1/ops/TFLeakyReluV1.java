package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.LeakyReluV1;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.nn.LeakyRelu;

public class TFLeakyReluV1 extends TFOperator implements LeakyReluV1<Tensor<?>> {

	@Override
	public Tensor<?> leakyRelu(Tensor<?> x, Float alpha, List<Long> consumedInputs) {
		Scope scope = new Scope(TFSession.get());
		return LeakyRelu.create(scope, TensorUtil.toConstant(scope, (Tensor<Number>) x), LeakyRelu.alpha(alpha))
				.asOutput().tensor();
	}

}
