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
package org.forwarder.backend.impls.tensorflow.opsets.v1;

import java.util.logging.Logger;

import org.forwarder.backend.impls.tensorflow.PerformanceTracker;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperatorTest;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFMaxPoolV1;
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