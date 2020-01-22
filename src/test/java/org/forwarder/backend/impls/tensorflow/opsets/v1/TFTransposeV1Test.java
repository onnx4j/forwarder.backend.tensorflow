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

import static org.junit.Assert.assertArrayEquals;

import java.nio.DoubleBuffer;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

import org.forwarder.backend.impls.tensorflow.PerformanceTracker;
import org.forwarder.backend.impls.tensorflow.opsets.TFOperatorTest;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFTransposeV1;
import org.junit.Test;
import org.tensorflow.Tensor;

import com.google.common.collect.Lists;
import com.google.common.primitives.Longs;

public class TFTransposeV1Test extends TFOperatorTest {

	private static Logger logger = Logger.getGlobal();

	@Test
	public void test1() throws Exception {
		this.testTranspose(
				new long[] { 2, 1, 3}, 
				Tensor.create(new long[] {1, 2, 3},  DoubleBuffer.allocate(1*2*3)), 
				Collections.unmodifiableList(Longs.asList(1, 0, 2))
			);
	}

	public void testTranspose(long[] excepted, Tensor<?> data, List<Long> perm) {
		TFTransposeV1 operator = new TFTransposeV1();
		PerformanceTracker tracker = PerformanceTracker.start();
		Tensor<?> y = operator.transpose(data, Lists.newArrayList(1L, 0L, 2L));
		logger.info(String.format("Total time: %sms", String.valueOf(tracker.stop())));
		assertArrayEquals(excepted, y.shape());
	}

}