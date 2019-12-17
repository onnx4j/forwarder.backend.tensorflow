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

import java.nio.ByteBuffer;

import org.tensorflow.Tensor;

import com.google.common.primitives.Floats;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for some performance tests.
 */
public class PerformanceTest extends TestCase {

	public PerformanceTest(String testName) {
		super(testName);
	}

	public static Test suite() {
		return new TestSuite(PerformanceTest.class);
	}

	public void testHeapByteBufferVsDirectedByteBuffer() {
		int[] runTimes = new int[] { 1, 10, 100, 1000, 10000, 100000 };

		//
		// Tensor size = 100M
		//
		int tensorSize = 1024 * 1024 * 100;

		byte[] byteOfTensor = new byte[tensorSize];

		//
		// Writes random data into byteOfTensor object
		//
		for (int n = 0; n < tensorSize; n++) {
			byteOfTensor[n] = '1';
		}
		System.out.println("Data array init done.");

		ByteBuffer heapByteBuffer = ByteBuffer.wrap(byteOfTensor);

		ByteBuffer directedByteBuffer = ByteBuffer.allocateDirect(tensorSize);
		directedByteBuffer.put(byteOfTensor);

		for (int n = 0; n < runTimes.length; n++) {
			this.execCeateTensorFromByteBuffer(runTimes[n], tensorSize, heapByteBuffer);
			this.execCeateTensorFromByteBuffer(runTimes[n], tensorSize, directedByteBuffer);
		}
	}

	private float execCeateTensorFromByteBuffer(int runTimes, int tensorSize, ByteBuffer buffer) {
		long startTimeMS = System.currentTimeMillis();
		for (int n = 0; n < runTimes; n++) {
			buffer.rewind();

			try (Tensor<Float> ts = org.tensorflow.Tensor.create(Float.class, new long[] { tensorSize / Floats.BYTES },
					buffer)) {
				assertEquals(tensorSize / Floats.BYTES, ts.numElements());
			}
		}
		long endTimeMS = System.currentTimeMillis();
		float usedTimems = (endTimeMS - startTimeMS);
		System.out.println(
				String.format("[%s]\tRun times: %s;\tTotal time: %ss;\tAvg time: %sms", buffer.getClass().getName(),
						runTimes, String.valueOf(usedTimems / 1000), String.valueOf(usedTimems / runTimes)));
		return usedTimems;
	}

}