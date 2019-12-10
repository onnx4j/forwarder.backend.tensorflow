package org.forwarder.backend.impls.tensorflow.opsets.v1;

import java.util.logging.Logger;

import org.forwarder.backend.impls.tensorflow.PerformanceTracker;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperatorTest;
import org.forwarder.backend.impls.tensorflow.opsets.v1.ops.TFMaxPoolV1;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.tensorflow.Tensor;

import junit.framework.Test;
import junit.framework.TestSuite;

public class TFMaxPoolV1Test extends TFOperatorTest {

	private static Logger logger = Logger.getGlobal();

	public TFMaxPoolV1Test(String testName) {
		super(testName);
	}

	public static Test suite() {
		return new TestSuite(TFMaxPoolV1Test.class);
	}

	public void testMaxPoolV1() {
//		Float[][][][] 4dArray = new Float[] {
//				[
//				    [[1.0, 2.0, 3.0, 4.0],
//				     [5.0, 6.0, 7.0, 8.0],
//				     [8.0, 7.0, 6.0, 5.0],
//				     [4.0, 3.0, 2.0, 1.0]],
//				    [[4.0, 3.0, 2.0, 1.0],
//				     [8.0, 7.0, 6.0, 5.0],
//				     [1.0, 2.0, 3.0, 4.0],
//				     [5.0, 6.0, 7.0, 8.0]]
//				]
//		}
		
		Tensor<Float> tensorA = TensorUtil.create(new Float[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f }, 4,
				3);
		Tensor<Float> tensorB = TensorUtil.create(new Float[] { 1f, 2f, 3f }, 1, 3);
		
		TFMaxPoolV1 operator = new TFMaxPoolV1();

		PerformanceTracker tracker = PerformanceTracker.start();
		Tensor<?> tensorC = null;//operator.maxpool(tensorA, tensorB, null, 0L, null);
		logger.info(String.format("Total time: %sms", String.valueOf(tracker.stop())));

		//logger.info(DumpUtil.dump(TensorUtil.copyOut(tensorC).asFloatBuffer(), tensorC.shape()));
	}

}