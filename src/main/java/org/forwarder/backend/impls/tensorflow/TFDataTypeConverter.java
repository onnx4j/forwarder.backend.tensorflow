/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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