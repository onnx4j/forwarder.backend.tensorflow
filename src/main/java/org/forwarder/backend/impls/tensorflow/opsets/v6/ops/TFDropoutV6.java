package org.forwarder.backend.impls.tensorflow.opsets.v6.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFDropoutV1;
import org.onnx4j.opsets.v6.ops.DropoutV6;
import org.tensorflow.Tensor;

public class TFDropoutV6 extends TFDropoutV1 implements DropoutV6<Tensor<?>> {

	@Override
	public List<Tensor<?>> dropout(Tensor<?> data, Boolean isTest, Float ratio) {
		return super.dropout(data, isTest, ratio, null);
	}

}
