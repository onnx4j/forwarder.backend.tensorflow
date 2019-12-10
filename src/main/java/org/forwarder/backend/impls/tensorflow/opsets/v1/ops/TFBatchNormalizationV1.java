package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.BatchNormalizationV1;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.linalg.Transpose;
import org.tensorflow.op.nn.FusedBatchNorm;

public class TFBatchNormalizationV1 extends TFOperator implements BatchNormalizationV1<Tensor<?>> {

	@Override
	public Tensor<?>[] batchNormalization(Tensor<?> x, Tensor<?> scale, Tensor<?> b, Tensor<?> mean, Tensor<?> var,
			List<Long> consumedInputs, Float epsilon, Boolean isTest, Float momentum, Boolean spatial) {
		Scope scope = new Scope(TFSession.get());
		
		Operand opX = TensorUtil.toConstant(scope, x);
		Operand opMean = TensorUtil.toConstant(scope, mean);
		Operand opVar = TensorUtil.toConstant(scope, var);
		Operand opBeta = TensorUtil.toConstant(scope, b);
		Operand opGamma = TensorUtil.toConstant(scope, scale);
		Output output = FusedBatchNorm.create(
				scope, 
				this.toNHWC(scope, opX),
				opGamma, 
				opBeta, 
				opMean, 
				opVar, 
				FusedBatchNorm
					.dataFormat("NHWC")
					.epsilon(epsilon)
					.isTraining(false)
				).y();
		/*Operation opBatchNorm = scope.env()
			.opBuilder("tf.nn.batch_normalization", scope.makeOpName("BatchNormWithGlobalNormalization"))
			.addInput(TensorUtil.toConstant(scope, x).asOutput())
			.addInput(TensorUtil.toConstant(scope, mean).asOutput())
			.addInput(TensorUtil.toConstant(scope, var).asOutput())
			.addInput(TensorUtil.toConstant(scope, b).asOutput())
			.addInput(TensorUtil.toConstant(scope, scale).asOutput())
			.setAttr("variance_epsilon", epsilon)
		    .setAttr("scale_after_normalization", true)
			.build();*/
		//Operand opBatchNorm = BatchNormWithGlobalNormalization.create(scope, opX, opMean, opVar, opBeta, opGamma,
		//		epsilon, true);
		return new Tensor<?>[] { this.toNCHW(scope, output).asOutput().tensor() };
	}

	private <T> Operand<T> toNCHW(Scope scope, Operand<T> inputNHWC) {
		return Transpose.create(scope, inputNHWC, Constant.create(scope, new int[] { 0, 3, 1, 2 }));
	}

	private <T> Operand<T> toNHWC(Scope scope, Operand<T> inputNCHW) {
		return Transpose.create(scope, inputNCHW, Constant.create(scope, new int[] { 0, 2, 3, 1 }));
	}

}