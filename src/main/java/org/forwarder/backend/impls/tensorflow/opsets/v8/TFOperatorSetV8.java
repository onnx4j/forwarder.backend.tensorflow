package org.forwarder.backend.impls.tensorflow.opsets.v8;

import org.forwarder.backend.impls.tensorflow.opsets.v7.TFOperatorSetV7;
import org.onnx4j.opsets.v8.OperatorSetSpecV8;
import org.tensorflow.Tensor;

public class TFOperatorSetV8 extends TFOperatorSetV7 implements OperatorSetSpecV8<Tensor<?>> {

	public TFOperatorSetV8() {
		super(1, "", "", "", 8L, "ONNX OPSET-V8 USING TENSORFLOW BACKEND");
	}

	public TFOperatorSetV8(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
