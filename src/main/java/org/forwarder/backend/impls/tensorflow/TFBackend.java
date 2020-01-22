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

import org.forwarder.Backend;
import org.forwarder.Session;
import org.forwarder.executor.Executor;
import org.onnx4j.Model;
import org.onnx4j.TensorManager;
import org.onnx4j.opsets.OperatorSetId;
import org.onnx4j.tensor.Shape;
import org.onnx4j.tensor.TensorBuilder;
import org.tensorflow.Tensor;

public class TFBackend extends Backend<Tensor<?>> {
	
	static {
		System.setProperty("org.tensorflow.NativeLibrary.DEBUG", "TRUE");
	}

	public static final String BACKEND_NAME = "Tensorflow";
	
	public TFBackend() { super(); }
	
	public TFBackend(Model model, Executor<Tensor<?>> executor) {
		super(model, executor);
	}
	
	public TFBackend(OperatorSetId[] opsetIds, Executor<Tensor<?>> executor) {
		super(opsetIds, executor);
	}

	@Override
	public String getName() { return BACKEND_NAME; }

	@Override
	public Session<Tensor<?>> newSession() {
		return new TFSession(super.getExecutor(), this);
	}

	@Override
	public void disposeBackendTensor(Tensor<?> backendTensor) {
		backendTensor.close();
	}

	@Override
	public Tensor<?> toBackendTensor(TensorManager<Tensor<?>> tensorManager, org.onnx4j.Tensor rawTensor) {
		Tensor<?> backendTensor = Tensor.create(
				rawTensor.getDataType().getPrototype(), 
				rawTensor.getShape(), 
				rawTensor.getData()
			);
		tensorManager.attach(rawTensor.getName(), backendTensor);
		return backendTensor;
	}

	@Override
	public org.onnx4j.Tensor toNativeTensor(TensorManager<org.onnx4j.Tensor> tensorManager, String name,
			Tensor<?> backendTensor) {
		org.onnx4j.Tensor rawTensor = TensorBuilder
				.builder(
					TFDataTypeConverter.toOnnx4jDataType(backendTensor.dataType()), 
					Shape.create(backendTensor.shape()),
					this.getModel().getTensorOptions()
				)
				.manager(tensorManager)
				.name(name)
				.docString("Created from org.tensorflow.Tensor<?> in TFBackend.toTensor()")
				.write(b -> backendTensor.writeTo(b))
				.build();
		return rawTensor;
	}

}