package org.forwarder.backend.impls.tensorflow;

import org.forwarder.Backend;
import org.forwarder.Model;
import org.forwarder.Session;
import org.onnx4j.opsets.OperatorSetId;
import org.onnx4j.tensor.Shape;
import org.tensorflow.Tensor;

public class TFBackend extends Backend<Tensor<?>> {
	
	static {
		System.setProperty("org.tensorflow.NativeLibrary.DEBUG", "TRUE");
	}

	public static final String BACKEND_NAME = "Tensorflow";
	
	public TFBackend() { super(); }
	
	public TFBackend(Model model) {
		super(model);
	}
	
	public TFBackend(OperatorSetId[] opsetIds) {
		super(opsetIds);
	}

	@Override
	public String getName() { return BACKEND_NAME; }

	@Override
	public Session<Tensor<?>> newSession() {
		return new TFSession(this);
	}

	@Override
	public void disposeBackendTensor(Tensor<?> backendTensor) {
		backendTensor.close();
	}

	@Override
	public Tensor<?> toBackendTensor(org.onnx4j.Tensor rawTensor) {
		Tensor<?> backendTensor = Tensor.create(
				rawTensor.getDataType().getPrototype(), 
				rawTensor.getShape(), 
				rawTensor.getData()
			);
		return backendTensor;
	}

	@Override
	public org.onnx4j.Tensor toTensor(Tensor<?> backendTensor) {
		org.onnx4j.Tensor rawTensor = org.onnx4j.Tensor
				.builder(
					TFDataTypeConverter.toOnnx4jDataType(backendTensor.dataType()), 
					Shape.create(backendTensor.shape()),
					this.getModel().getTensorOptions()
				)
				.write(b -> backendTensor.writeTo(b))
				.build();
		/*org.onnx4j.Tensor rawTensor = new org.onnx4j.Tensor(
				TFDataTypeConverter.toOnnx4jDataType(backendTensor.dataType()), 
				Shape.create(backendTensor.shape()), 
				Memory.builder(backendTensor.numBytes()).write(b -> backendTensor.writeTo(b)).build());*/
		return rawTensor;
	}

}
