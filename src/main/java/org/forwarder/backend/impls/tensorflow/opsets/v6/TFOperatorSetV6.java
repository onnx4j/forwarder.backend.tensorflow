package org.forwarder.backend.impls.tensorflow.opsets.v6;

import org.forwarder.backend.impls.tensorflow.opsets.v5.TFOperatorSetV5;
import org.forwarder.backend.impls.tensorflow.opsets.v6.ops.TFDropoutV6;
import org.forwarder.backend.impls.tensorflow.opsets.v6.ops.TFMulV6;
import org.onnx4j.opsets.v6.OperatorSetSpecV6;
import org.onnx4j.opsets.v6.ops.DropoutV6;
import org.onnx4j.opsets.v6.ops.MulV6;
import org.tensorflow.Tensor;

public class TFOperatorSetV6 extends TFOperatorSetV5 implements OperatorSetSpecV6<Tensor<?>> {

	@Override
	public MulV6<Tensor<?>> getMulV6() { return new TFMulV6(); }

	@Override
	public DropoutV6<Tensor<?>> getDropoutV6() { return new TFDropoutV6(); }

	public TFOperatorSetV6() {
		super(1, "", "", "", 6L, "ONNX OPSET-V6 USING TENSORFLOW BACKEND");
	}

	public TFOperatorSetV6(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
