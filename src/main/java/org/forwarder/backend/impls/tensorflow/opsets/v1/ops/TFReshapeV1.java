package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.ReshapeV1;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Reshape;

import com.google.common.primitives.Longs;

public class TFReshapeV1 extends TFOperator implements ReshapeV1<Tensor<?>> {

	@Override
	public Tensor<?> reshape(Tensor<?> a, List<Long> shape, List<Long> consumedInputs) {
		Scope scope = new Scope(TFSession.get());
		return Reshape.create(scope, 
				TensorUtil.toConstant(scope, a), 
				Constant.create(scope, Longs.toArray(shape))
				).asOutput().tensor();
	}

}
