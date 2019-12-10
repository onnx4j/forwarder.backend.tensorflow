package org.forwarder.backend.impls.tensorflow;

import org.tensorflow.DataType;

public final class TFDataTypeConverter {
	
	public static org.onnx4j.tensor.DataType toOnnx4jDataType(DataType dataType) {
		if (org.tensorflow.DataType.FLOAT == dataType)
			return org.onnx4j.tensor.DataType.FLOAT;
		else if (org.tensorflow.DataType.BOOL == dataType)
			return org.onnx4j.tensor.DataType.BOOL;
		else if (org.tensorflow.DataType.DOUBLE == dataType)
			return org.onnx4j.tensor.DataType.DOUBLE;
		else if (org.tensorflow.DataType.INT32 == dataType)
			return org.onnx4j.tensor.DataType.INT32;
		else if (org.tensorflow.DataType.INT64 == dataType)
			return org.onnx4j.tensor.DataType.INT64;
		else if (org.tensorflow.DataType.STRING == dataType)
			return org.onnx4j.tensor.DataType.STRING;
		else if (org.tensorflow.DataType.UINT8 == dataType)
			return org.onnx4j.tensor.DataType.UINT8;
		else
			return null;
	}

}
