package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.LinkedList;
import java.util.List;

import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.ConvV1;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.linalg.Transpose;
import org.tensorflow.op.nn.BiasAdd;
import org.tensorflow.op.nn.Conv2d;

import com.google.common.collect.Lists;

public class TFConvV1 extends TFOperator implements ConvV1<Tensor<?>> {

	@Override
	public Tensor<?> conv(Tensor<?> x, Tensor<?> w, Tensor<?> b, String autoPad, List<Long> dilations, Long group,
			List<Long> kernelShape, List<Long> pads, List<Long> strides) {
		Scope scope = new Scope(TFSession.get());
		
		//
		// Translate inputs from (N x C x KH x KW) to (N X KH x KW X C)
		// Tensorflow for java does not support NCHW mode on CPU temporarily
		//
		Operand<Number> operandX = this.toNHWC(scope, TensorUtil.toConstant(scope, (Tensor<Number>) x));
		
		//
		// Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
		//
		Operand<Number> operandW = this.toHWCN(scope, TensorUtil.toConstant(scope, (Tensor<Number>) w));

		//
		// Add 1L to first and last of dilations=[m, n]
		// =>
		// [1, m, n, 1]
		//
		LinkedList<Long> newDilations = Lists.newLinkedList(dilations);
		newDilations.addFirst(1L);
		newDilations.addLast(1L);

		//
		// Add 1L to first and last of strides=[m, n]
		// =>
		// [1, m, n, 1]
		//
		LinkedList<Long> newStrides = Lists.newLinkedList(strides);
		newStrides.addFirst(1L);
		newStrides.addLast(1L);

		Conv2d.Options options = Conv2d
				.useCudnnOnGpu(false)
				.dataFormat("NHWC")
				.dilations(newDilations);
		Operand<Number> opreandConv2D = Conv2d.create(
				scope, 
				operandX, 
				operandW, 
				newStrides,
				this.getTFPadding(x, w, b, autoPad, dilations, group, kernelShape, pads, strides), 
				options);
		
		if (b == null) {
			return this.toNCHW(scope, opreandConv2D).asOutput().tensor();
		} else {
			Operand opB = TensorUtil.toConstant(scope, b);
			return this.toNCHW(scope, BiasAdd.create(scope, opreandConv2D, opB, BiasAdd.dataFormat("NHWC"))).asOutput().tensor();
			/*throw new NotImplementedException(
					String.format("[%s] Tensorflow can not handle bias data", MaxPoolV1.OP_TYPE));*/
		}
	}

	private String getTFPadding(Tensor<?> x, Tensor<?> w, Tensor<?> b, String autoPad, List<Long> dilations, Long group,
			List<Long> kernelShape, List<Long> pads, List<Long> strides) {
		if ("VALID".equalsIgnoreCase(autoPad))
			return "VALID";
		else if ("SAME_UPPER".equalsIgnoreCase(autoPad) || "SAME_LOWER".equalsIgnoreCase(autoPad))
			return "SAME";
		else
			throw new RuntimeException(String.format("Tensorflow can not support \"%s\" padding mode", autoPad));
	}

	private <T> Operand<T> toNCHW(Scope scope, Operand<T> inputNHWC) {
		return Transpose.create(scope, inputNHWC, Constant.create(scope, new int[] { 0, 3, 1, 2 }));
	}

	private <T> Operand<T> toNHWC(Scope scope, Operand<T> inputNCHW) {
		return Transpose.create(scope, inputNCHW, Constant.create(scope, new int[] { 0, 2, 3, 1 }));
	}

	private <T> Operand<T> toHWCN(Scope scope, Operand<T> inputNCHW) {
		return Transpose.create(scope, inputNCHW, Constant.create(scope, new int[] { 2, 3, 1, 0 }));
	}

}
