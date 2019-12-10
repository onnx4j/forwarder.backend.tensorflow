package org.forwarder.backend.impls.tensorflow.opsets.v1.ops;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.forwarder.backend.impls.tensorflow.TFSession;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperator;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.onnx4j.opsets.v1.ops.MaxPoolV1;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.linalg.Transpose;
import org.tensorflow.op.nn.MaxPool;

import com.google.common.collect.Lists;
import com.google.common.primitives.Ints;

public class TFMaxPoolV1 extends TFOperator implements MaxPoolV1<Tensor<?>> {

	@Override
	public Tensor<?> maxpool(Tensor<?> data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides) {
		Scope scope = new Scope(TFSession.get());

		Operand<?> constantData = TensorUtil.toConstant(scope, data);

		//
		// Add 1L to first and last of kernelShape=[m, n]
		// =>
		// [1, m, n, 1]
		//
		LinkedList<Long> newKernelShape = Lists.newLinkedList(kernelShape);
		newKernelShape.addFirst(1L);
		newKernelShape.addLast(1L);

		//
		// Add 1L to first and last of strides=[m, n]
		// =>
		// [1, m, n, 1]
		//
		LinkedList<Long> newStrides = Lists.newLinkedList(strides);
		newStrides.addFirst(1L);
		newStrides.addLast(1L);

		Operand<?> opMaxPool = MaxPool.create(
				scope, 
				this.toNHWC(scope, constantData), 
				Constant.create(scope, Ints.toArray(newKernelShape)), 
				Constant.create(scope, Ints.toArray(newStrides)),
				this.getTFPadding(data, autoPad, kernelShape, pads, strides), 
				MaxPool.dataFormat("NHWC"));
		return this.toNCHW(scope, opMaxPool).asOutput().tensor();
	}

	private String getTFPadding(Tensor<?> data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides) {
		if ("NOTSET".equalsIgnoreCase(autoPad)) {
			if (pads != null && pads.size() > 0) {
				int nbSpatialSize = kernelShape.size();
				Long[] validPads = new Long[nbSpatialSize * 2];
				Arrays.fill(validPads, 0, validPads.length, 0L);
				
				if (Arrays.deepEquals(validPads, pads.toArray(new Long[pads.size()])))
					return "VALID";
				
				throw new NotImplementedException(String.format("[%s] Tensorflow can not support \"%s\" padding mode", 
						MaxPoolV1.OP_TYPE, autoPad));
			} else {
				return "VALID";
			}
		} else {
			if ("SAME_UPPER".equalsIgnoreCase(autoPad) || "SAME_LOWER".equalsIgnoreCase(autoPad))
				return "SAME";
			else if ("VALID".equalsIgnoreCase(autoPad))
				return "VALID";
			else
				throw new NotImplementedException(String.format("[%s] Tensorflow can not support \"%s\" padding mode", 
						MaxPoolV1.OP_TYPE, autoPad));
		}
	}

	private <T> Operand<T> toNCHW(Scope scope, Operand<T> inputNHWC) {
		return Transpose.create(scope, inputNHWC, Constant.create(scope, new int[] { 0, 3, 1, 2 }));
	}

	private <T> Operand<T> toNHWC(Scope scope, Operand<T> inputNCHW) {
		return Transpose.create(scope, inputNCHW, Constant.create(scope, new int[] { 0, 2, 3, 1 }));
	}

}
