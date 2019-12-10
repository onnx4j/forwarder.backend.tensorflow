package org.forwarder.backend.impls.tensorflow.opsets.v4.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFConcatV1;
import org.onnx4j.opsets.v4.ops.ConcatV4;
import org.tensorflow.Tensor;

public class TFConcatV4 extends TFConcatV1 implements ConcatV4<Tensor<?>> {

	@Override
	public Tensor<?> concat(List<Tensor<?>> inputs, Long axis) {
		return super.concat(inputs, axis);
	}

}
