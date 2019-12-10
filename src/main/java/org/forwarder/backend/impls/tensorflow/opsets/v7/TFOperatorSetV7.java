package org.forwarder.backend.impls.tensorflow.opsets.v7;

import org.forwarder.backend.impls.tensorflow.opsets.v6.TFOperatorSetV6;
import org.forwarder.backend.impls.tensorflow.opsets.v7.ops.TFAveragePoolV7;
import org.forwarder.backend.impls.tensorflow.opsets.v7.ops.TFBatchNormalizationV7;
import org.forwarder.backend.impls.tensorflow.opsets.v7.ops.TFDropoutV7;
import org.onnx4j.opsets.v7.OperatorSetSpecV7;
import org.onnx4j.opsets.v7.ops.AveragePoolV7;
import org.onnx4j.opsets.v7.ops.BatchNormalizationV7;
import org.onnx4j.opsets.v7.ops.DropoutV7;
import org.tensorflow.Tensor;

public class TFOperatorSetV7 extends TFOperatorSetV6 implements OperatorSetSpecV7<Tensor<?>> {

	@Override
	public BatchNormalizationV7<Tensor<?>> getBatchNormalizationV7() { return new TFBatchNormalizationV7(); }

	@Override
	public AveragePoolV7<Tensor<?>> getAveragePoolV7() { return new TFAveragePoolV7(); }

	@Override
	public DropoutV7<Tensor<?>> getDropoutV7() { return new TFDropoutV7(); }

	public TFOperatorSetV7() {
		super(1, "", "", "", 7L, "ONNX OPSET-V7 USING TENSORFLOW BACKEND");
	}

	public TFOperatorSetV7(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
