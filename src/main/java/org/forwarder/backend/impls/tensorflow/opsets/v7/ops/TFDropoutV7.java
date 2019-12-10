package org.forwarder.backend.impls.tensorflow.opsets.v7.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.opsets.v6.ops.TFDropoutV6;
import org.onnx4j.opsets.v7.ops.DropoutV7;
import org.tensorflow.Tensor;

public class TFDropoutV7 extends TFDropoutV6 implements DropoutV7<Tensor<?>> {

	@Override	
	public List<Tensor<?>> dropout(Tensor<?> data, Float ratio) {
		return super.dropout(data, true, ratio, null);
	}

}
