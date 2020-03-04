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
package org.forwarder.backend.impls.tensorflow.utils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.List;

import org.tensorflow.Operation;
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

	public static <T> Operation toConstant(Scope scope, Tensor<T> tensor) {
		return scope.env()
				.opBuilder("Const", scope.makeOpName("Const"))
				.setAttr("dtype", tensor.dataType())
				.setAttr("value", tensor)
				.build();
	}
	
	public static <T> List<Operation> toConstant(Scope scope, List<Tensor<T>> tensors) {
		List<Operation> constants = new LinkedList<Operation>();
		for (Tensor<T> tensor : tensors) {
			constants.add(toConstant(scope, tensor));
		}
		return constants;
	}

}