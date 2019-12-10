package org.forwarder.backend.impls.tensorflow.opsets.v2;

import org.forwarder.backend.impls.tensorflow.opsets.v1.TFOperatorSetV1;
import org.onnx4j.opsets.v2.OperatorSetSpecV2;
import org.tensorflow.Tensor;

public class TFOperatorSetV2 extends TFOperatorSetV1 implements OperatorSetSpecV2<Tensor<?>> {

	public TFOperatorSetV2() {
		this(1, "", "", "", 2L, "ONNX OPSET-V2 USING TENSORFLOW BACKEND");
	}

	public TFOperatorSetV2(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
