package org.forwarder.backend.impls.tensorflow.opsets.v7.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFBatchNormalizationV1;
import org.onnx4j.opsets.v7.ops.BatchNormalizationV7;
import org.tensorflow.Tensor;

public class TFBatchNormalizationV7 extends TFBatchNormalizationV1 implements BatchNormalizationV7<Tensor<?>> {

	@Override
	public Tensor<?>[] batchNormalization(Tensor<?> x, Tensor<?> scale, Tensor<?> b, Tensor<?> mean, Tensor<?> var,
			List<Long> consumedInputs, Float epsilon, Float momentum, Boolean spatial) {
		return super.batchNormalization(x, scale, b, mean, var, consumedInputs, epsilon, true, momentum, spatial);
	}

}
