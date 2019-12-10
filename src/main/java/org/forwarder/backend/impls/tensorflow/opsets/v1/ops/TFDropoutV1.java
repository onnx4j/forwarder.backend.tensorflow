package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.List;
import java.util.Optional;

import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.onnx4j.opsets.v1.ops.DropoutV1;
import org.tensorflow.Tensor;

public class TFDropoutV1 extends TFOperator implements DropoutV1<Tensor<?>> {

	@Override
	public List<Tensor<?>> dropout(Tensor<?> data, Boolean isTest, Float ratio, List<Long> consumedInputs) {
		if (isTest == false)
			throw new UnsupportedOperationException("Can not run " + DropoutV1.OP_TYPE + " in not test mode");
		
		return this.wrapMultiOutputs(Optional.of(data), Optional.empty());
	}

}
