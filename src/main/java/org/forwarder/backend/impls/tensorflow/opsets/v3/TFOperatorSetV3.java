package org.forwarder.backend.impls.tensorflow.opsets.v3;

import org.forwarder.backend.impls.tensorflow.opsets.v2.TFOperatorSetV2;
import org.onnx4j.opsets.v3.OperatorSetSpecV3;
import org.tensorflow.Tensor;

public class TFOperatorSetV3 extends TFOperatorSetV2 implements OperatorSetSpecV3<Tensor<?>> {

	public TFOperatorSetV3() {
		super(1, "", "", "", 3L, "ONNX OPSET-V3 USING TENSORFLOW BACKEND");
	}

	public TFOperatorSetV3(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
