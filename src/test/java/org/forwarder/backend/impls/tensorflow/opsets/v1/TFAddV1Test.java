package org.forwarder.backend.impls.tensorflow.opsets.v1;

import java.util.logging.Logger;

import org.forwarder.backend.impls.tensorflow.PerformanceTracker;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperatorTest;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFAddV1;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.tensorflow.Tensor;

import junit.framework.Test;
import junit.framework.TestSuite;

public class TFAddV1Test extends TFOperatorTest {

	private static Logger logger = Logger.getGlobal();

	public TFAddV1Test(String testName) {
		super(testName);
	}

	public static Test suite() {
		return new TestSuite(TFAddV1Test.class);
	}

	public void testAddV1() {
		Tensor<Float> tensorA = TensorUtil.create(new Float[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f }, 4,
				3);
		Tensor<Float> tensorB = TensorUtil.create(new Float[] { 1f, 2f, 3f }, 1, 3);
		
		TFAddV1 operator = new TFAddV1();

		PerformanceTracker tracker = PerformanceTracker.start();
		Tensor<?> tensorC = operator.add(tensorA, tensorB, null, 0L, null);
		logger.info(String.format("Total time: %sms", String.valueOf(tracker.stop())));

		//logger.info(DumpUtil.dump(TensorUtil.copyOut(tensorC).asFloatBuffer(), tensorC.shape()));
	}

}
