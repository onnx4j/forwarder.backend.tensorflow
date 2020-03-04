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

import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.forwarder.backend.impls.tensorflow.opsets.TFOperatorTest;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFTransposeV1;
import org.junit.Test;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import com.google.common.base.Stopwatch;
import com.google.common.primitives.Longs;

public class TFTransposeV1Test extends TFOperatorTest {

	private static Logger logger = Logger.getGlobal();

	@Test
	public void test1() throws Exception {
		this.testTranspose(new long[] { 2, 1, 3 }, Tensors.create(new long[] { 1, 2, 3 }),
				Collections.unmodifiableList(Longs.asList(1, 0, 2)));
	}

	public void testTranspose(long[] excepted, Tensor<? extends Number> data, List<Long> perm) {
		Stopwatch watch = Stopwatch.createStarted();
		Tensor<? extends Number> y = new TFTransposeV1() {

			@Override
			public Tensor<? extends Number> transpose(Tensor<? extends Number> data, List<Long> perm) {
				return super.transpose(data, perm);
			}

		}.transpose(data, perm);
		logger.info(String.format("Total time: %sms", String.valueOf(watch.elapsed(TimeUnit.MILLISECONDS))));
		assertArrayEquals(excepted, y.shape());
	}

}