package org.forwarder.backend.impls.tensorflow.opsets.v5;

import org.forwarder.backend.impls.tensorflow.opsets.v4.TFOperatorSetV4;
import org.forwarder.backend.impls.tensorflow.opsets.v5.ops.TFReshapeV5;
import org.onnx4j.opsets.v5.OperatorSetSpecV5;
import org.onnx4j.opsets.v5.ops.ReshapeV5;
import org.tensorflow.Tensor;

public class TFOperatorSetV5 extends TFOperatorSetV4 implements OperatorSetSpecV5<Tensor<?>> {

	@Override
	public ReshapeV5<Tensor<?>> getReshapeV5() { return new TFReshapeV5(); }

	public TFOperatorSetV5() {
		super(1, "", "", "", 5L, "ONNX OPSET-V7 USING TENSORFLOW BACKEND");
	}

	public TFOperatorSetV5(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
