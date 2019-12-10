package org.forwarder.backend.impls.tensorflow.opsets.v4;

import org.forwarder.backend.impls.tensorflow.opsets.v3.TFOperatorSetV3;
import org.forwarder.backend.impls.tensorflow.opsets.v4.ops.TFConcatV4;
import org.onnx4j.opsets.v4.OperatorSetSpecV4;
import org.onnx4j.opsets.v4.ops.ConcatV4;
import org.tensorflow.Tensor;

public class TFOperatorSetV4 extends TFOperatorSetV3 implements OperatorSetSpecV4<Tensor<?>> {

	@Override
	public ConcatV4<Tensor<?>> getConcatV4() { return new TFConcatV4(); }

	public TFOperatorSetV4() {
		super(1, "", "", "", 4L, "ONNX OPSET-V7 USING TENSORFLOW BACKEND");
	}

	public TFOperatorSetV4(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
