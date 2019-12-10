package org.forwarder.backend.impls.tensorflow.opsets.v6.ops;

import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFMulV1;
import org.onnx4j.opsets.v6.ops.MulV6;
import org.tensorflow.Tensor;

public class TFMulV6 extends TFMulV1 implements MulV6<Tensor<?>> {

	@Override
	public Tensor<?> mul(Tensor<?> a, Tensor<?> b, Long axis, Long broadcast) {
		return super.mul(a, b, axis, broadcast, null);
	}

}
