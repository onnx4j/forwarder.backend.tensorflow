package org.forwarder.backend.impls.tensorflow;

import static org.junit.Assert.assertEquals;

import java.io.UnsupportedEncodingException;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.Random;

import org.junit.Test;
import org.tensorflow.EagerSession;
import org.tensorflow.EagerSession.DevicePlacementPolicy;
import org.tensorflow.EagerSession.ResourceCleanupStrategy;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.Tensors;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;
import org.tensorflow.op.Scope;

public class TensorflowTest {
	
	static {
		System.setProperty("org.tensorflow.NativeLibrary.DEBUG", "3");
	}
	
	@Test
	public void testHelloWorld() throws UnsupportedEncodingException {
		try (Graph g = new Graph()) {
	        final String value = "Hello from " + TensorFlow.version();

	        // Construct the computation graph with a single operation, a constant
	        // named "MyConst" with a value "value".
	        try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
	            // The Java API doesn't yet include convenience functions for adding operations.
	            g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
	        }

	        // Execute the "MyConst" operation in a Session.
	        try (Session s = new Session(g);
	                Tensor output = s.runner().fetch("MyConst").run().get(0)) {
	            System.out.println(new String(output.bytesValue(), "UTF-8"));
	        }
	    }
	}
	
	@Test
	public void testHelloWorldWithGPU() throws UnsupportedEncodingException {
		final String device = "/gpu:0";
		byte[] configProto = ConfigProto.newBuilder()
				.setGpuOptions(GPUOptions.newBuilder()
						.setAllowGrowth(true)
						.setForceGpuCompatible(true)
						.setPerProcessGpuMemoryFraction(0.1d)
						)
				.setAllowSoftPlacement(false)
				.setLogDevicePlacement(false)
				.build()
				.toByteArray();
		Random rnd = new Random();
		
		for (int n = 0; n < 10000; n++) {
			double[] data = new double[1000];
			for (int i = 0; i < 1000; i++) {
				data[i] = rnd.nextDouble();
			}
			try (EagerSession s = EagerSession.options()
					.config(configProto)
					.devicePlacementPolicy(DevicePlacementPolicy.WARN)
					.resourceCleanupStrategy(ResourceCleanupStrategy.ON_SESSION_CLOSE)
					.build()
			) {
				Scope scope = new Scope(s);
		        try (Tensor<Double> a = Tensors.create(data); Tensor<Double> b = Tensors.create(data)) {
		        	Operand<Double> constA = scope.env()
		    				.opBuilder("Const", scope.makeOpName("Const"))
		    				.setAttr("dtype", a.dataType())
		    				.setAttr("value", a)
		    				.setDevice(device)
		    				.build()
		    				.output(0);
		        	Operand<Double> constB = scope.env()
		    				.opBuilder("Const", scope.makeOpName("Const"))
		    				.setAttr("dtype", b.dataType())
		    				.setAttr("value", b)
		    				.setDevice(device)
		    				.build()
		    				.output(0);
		        	Operand<Double> add = scope.env()
		        			.opBuilder("Add", scope.makeOpName("Add"))
		        			.addInput(constA.asOutput())
		        			.addInput(constB.asOutput())
		    				.setDevice(device)
		        			.build()
		        			.output(0);
		        	
		        	double[] output = new double[1000];
		        	add.asOutput().tensor().copyTo(output);
		        	for (int k = 0; k < 1000; k++)
		        		System.out.print(output[k] + ", ");
		        	
		        	System.out.print("\n");
		        }
		    }
		}
	}

}
