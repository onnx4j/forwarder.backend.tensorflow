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

import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.forwarder.backend.impls.tensorflow.PerformanceTracker;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperatorTest;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFReshapeV1;
import org.forwarder.backend.impls.tensorflow.utils.TensorUtil;
import org.tensorflow.Tensor;

import com.google.common.collect.Lists;

import junit.framework.Test;
import junit.framework.TestSuite;

public class TFReshape1Test extends TFOperatorTest {

	private static Logger logger = Logger.getGlobal();

	public TFReshape1Test(String testName) {
		super(testName);
	}

	public static Test suite() {
		return new TestSuite(TFReshape1Test.class);
	}

	public void testReshape1() {
		Tensor<Float> tensorA = TensorUtil.create(new Float[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f }, 4,
				3);
		List<Long> newShape = Lists.newArrayList(12L, 1L);
		
		TFReshapeV1 operator = new TFReshapeV1();

		PerformanceTracker tracker = PerformanceTracker.start();
		Tensor<?> outTensor = operator.reshape(tensorA, newShape, null);
		logger.info(String.format("Total time: %sms", String.valueOf(tracker.stop())));
		assertTrue(Arrays.equals(new long[] {12L, 1L},  outTensor.shape()));
	}

}