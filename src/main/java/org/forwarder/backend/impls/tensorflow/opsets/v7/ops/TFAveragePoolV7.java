package org.forwarder.backend.impls.tensorflow.opsets.v7.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFAveragePoolV1;
import org.onnx4j.opsets.v7.ops.AveragePoolV7;
import org.tensorflow.Tensor;

public class TFAveragePoolV7 extends TFAveragePoolV1 implements AveragePoolV7<Tensor<?>> {

	@Override
	public Tensor<?> averagePool(Tensor<?> data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides, Long countIncludePad) {
		if (countIncludePad != null && countIncludePad != 0L)
			throw new UnsupportedOperationException(
					String.format("[%s] Unable to handle \"countIncludePad\" is not equals to 0L", OP_TYPE));

		return super.averagePool(data, autoPad, kernelShape, pads, strides);
	}

}
