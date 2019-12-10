package org.forwarder.backend.impls.tensorflow.opsets.v1;

import org.forwarder.backend.impls.tensorflow.TFBackend;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperatorSet;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFAbsV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFAddV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFArgMaxV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFAveragePoolV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFBatchNormalizationV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFConcatV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFConstantV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFConvV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFDivV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFDropoutV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFIdentityV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFImageScalerV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFLeakyReluV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFMatMulV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFMaxPoolV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFMulV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFPadV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFReluV1;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFReshapeV1;
import org.forwarder.opset.annotations.Opset;
import org.onnx4j.opsets.v1.OperatorSetSpecV1;
import org.onnx4j.opsets.v1.ops.AbsV1;
import org.onnx4j.opsets.v1.ops.AddV1;
import org.onnx4j.opsets.v1.ops.ArgMaxV1;
import org.onnx4j.opsets.v1.ops.AveragePoolV1;
import org.onnx4j.opsets.v1.ops.BatchNormalizationV1;
import org.onnx4j.opsets.v1.ops.ConcatV1;
import org.onnx4j.opsets.v1.ops.ConstantV1;
import org.onnx4j.opsets.v1.ops.ConvV1;
import org.onnx4j.opsets.v1.ops.DivV1;
import org.onnx4j.opsets.v1.ops.DropoutV1;
import org.onnx4j.opsets.v1.ops.IdentityV1;
import org.onnx4j.opsets.v1.ops.ImageScalerV1;
import org.onnx4j.opsets.v1.ops.LeakyReluV1;
import org.onnx4j.opsets.v1.ops.MatMulV1;
import org.onnx4j.opsets.v1.ops.MaxPoolV1;
import org.onnx4j.opsets.v1.ops.MulV1;
import org.onnx4j.opsets.v1.ops.PadV1;
import org.onnx4j.opsets.v1.ops.ReluV1;
import org.onnx4j.opsets.v1.ops.ReshapeV1;
import org.tensorflow.Tensor;

@Opset(backendName = TFBackend.BACKEND_NAME)
public class TFOperatorSetV1 extends TFOperatorSet implements OperatorSetSpecV1<Tensor<?>> {

	@Override
	public AbsV1<Tensor<?>> getAbsV1() { return new TFAbsV1(); }

	@Override
	public PadV1<Tensor<?>> getPadV1() { return new TFPadV1(); }

	@Override
	public MatMulV1<Tensor<?>> getMatMulV1() { return new TFMatMulV1(); }

	@Override
	public IdentityV1<Tensor<?>> getIdentityV1() { return new TFIdentityV1(); }

	@Override
	public ArgMaxV1<Tensor<?>> getArgMaxV1() { return new TFArgMaxV1(); }

	@Override
	public DivV1<Tensor<?>> getDivV1() { return new TFDivV1(); }

	@Override
	public ReshapeV1<Tensor<?>> getReshapeV1() { return new TFReshapeV1(); }

	@Override
	public MaxPoolV1<Tensor<?>> getMaxPoolV1() { return new TFMaxPoolV1(); }

	@Override
	public AddV1<Tensor<?>> getAddV1() { return new TFAddV1(); }

	@Override
	public ConstantV1<Tensor<?>> getConstantV1() { return new TFConstantV1(); }

	@Override
	public ReluV1<Tensor<?>> getReluV1() { return new TFReluV1(); }

	@Override
	public ConvV1<Tensor<?>> getConvV1() { return new TFConvV1(); }

	@Override
	public ImageScalerV1<Tensor<?>> getImageScalerV1() { return new TFImageScalerV1(); }

	@Override
	public BatchNormalizationV1<Tensor<?>> getBatchNormalizationV1() { return new TFBatchNormalizationV1(); }

	@Override
	public LeakyReluV1<Tensor<?>> getLeakyReluV1() { return new TFLeakyReluV1(); }

	@Override
	public MulV1<Tensor<?>> getMulV1() { return new TFMulV1(); }

	@Override
	public ConcatV1<Tensor<?>> getConcatV1() { return new TFConcatV1(); }

	@Override
	public DropoutV1<Tensor<?>> getDropoutV1() { return new TFDropoutV1(); }

	@Override
	public AveragePoolV1<Tensor<?>> getAveragePoolV1() { return new TFAveragePoolV1(); }

	public TFOperatorSetV1() {
		this(1, "", "", "", 1L, "ONNX OPSET-V1 USING TENSORFLOW BACKEND");
	}

	public TFOperatorSetV1(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
