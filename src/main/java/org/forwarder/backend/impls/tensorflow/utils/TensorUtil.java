package org.forwarder.backend.impls.tensorflow.utils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;

import com.google.common.primitives.Floats;

public class TensorUtil {

	public static Tensor<Float> create(Float[] flatDataArray, long... shape) {
		ByteBuffer buffer = ByteBuffer.allocateDirect(flatDataArray.length * Floats.BYTES)
				.order(ByteOrder.nativeOrder());
		for (Float data : flatDataArray) {
			buffer.putFloat(data);
		}
		buffer.flip();
		return Tensor.create(Float.class, shape, buffer);
	}

	public static ByteBuffer copyOut(Tensor<?> tensor) {
		ByteBuffer buffer = ByteBuffer.allocate(tensor.numBytes()).order(ByteOrder.nativeOrder());
		tensor.writeTo(buffer);
		buffer.flip();
		return buffer;
	}
	
	public static <T> Operand<T> toConstant(Scope scope, Tensor<T> tensor) {
		return scope.env()
				.opBuilder("Const", scope.makeOpName("Const"))
				.setAttr("dtype", tensor.dataType())
				.setAttr("value", tensor)
				.build()
				.output(0);
	}
	
	public static <T> List<Operand<T>> toConstant(Scope scope, List<Tensor<T>> tensors) {
		List<Operand<T>> constants = new LinkedList<Operand<T>>();
		for (Tensor<T> tensor : tensors) {
			constants.add(toConstant(scope, tensor));
		}
		return constants;
	}

}
