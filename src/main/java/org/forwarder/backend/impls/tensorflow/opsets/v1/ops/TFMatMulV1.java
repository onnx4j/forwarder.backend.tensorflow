package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.MatMulV1;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.linalg.MatMul;

public class TFMatMulV1 extends TFOperator implements MatMulV1<Tensor<?>> {

	@Override
	public Tensor<?> matmul(Tensor<?> x0, Tensor<?> x1) {
		Scope scope = new Scope(TFSession.get());
		Operand<Object> opMatMul = MatMul.create(
				scope, (Operand<Object>) TensorUtil.toConstant(scope, x0),
				(Operand<Object>) TensorUtil.toConstant(scope, x1), 
				MatMul.transposeA(false).transposeB(false));
		return opMatMul.asOutput().tensor();
	}

}
