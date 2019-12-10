package org.forwarder.backend.impls.tensorflow.opsets.v5.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFReshapeV1;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v5.ops.ReshapeV5;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Reshape;

public class TFReshapeV5 extends TFReshapeV1 implements ReshapeV5<Tensor<?>> {

	@Override
	public Tensor<?> reshape(Tensor<?> data, Tensor<?> shape, List<Long> consumedInputs) {
		Scope scope = new Scope(TFSession.get());
		Operand<Number> opReshape = Reshape.create(scope, (Operand<Number>) TensorUtil.toConstant(scope, data),
				(Operand<Number>) TensorUtil.toConstant(scope, shape));
		return opReshape.asOutput().tensor();
	}

}
