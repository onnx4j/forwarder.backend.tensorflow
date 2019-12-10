package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.ImageScalerV1;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.math.Mul;

@Deprecated
public class TFImageScalerV1 extends TFOperator implements ImageScalerV1<Tensor<?>> {

	@Override
	public Tensor<?> scale(Tensor<?> input, Float scale, List<Float> bias) {
		Scope scope = new Scope(TFSession.get());
		
		Operand x = TensorUtil.toConstant(scope, input);
		Operand y = Constant.create(scope, scale);
		//BiasAdd.create(scope, value, bias, BiasAdd.dataFormat(dataFormat))
		return Mul.create(scope, x, y).asOutput().tensor();
	}

}
