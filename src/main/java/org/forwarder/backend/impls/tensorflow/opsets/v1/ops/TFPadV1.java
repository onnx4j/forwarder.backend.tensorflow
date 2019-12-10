package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.onnx4j.opsets.v1.ops.PadV1;
import org.tensorflow.Tensor;

public class TFPadV1 extends TFOperator implements PadV1<Tensor<?>> {

	@Override
	public Tensor<?> pad(Tensor<?> x) {
		return null;
	}

}
