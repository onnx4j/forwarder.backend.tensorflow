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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v8;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v7.TFAiOnnxOperatorSetV7;
import org.onnx4j.opsets.aiOnnx.v8.AiOnnxOperatorSetSpecV8;
import org.onnx4j.opsets.aiOnnx.v8.ops.SumV8;
import org.tensorflow.Tensor;

public class TFAiOnnxOperatorSetV8 extends TFAiOnnxOperatorSetV7 implements AiOnnxOperatorSetSpecV8<Tensor<?>> {

	public TFAiOnnxOperatorSetV8() {
		super(1, "", "", 8L, "ONNX OPSET-V8 USING TENSORFLOW BACKEND");
	}

	public TFAiOnnxOperatorSetV8(int irVersion, String irVersionPrerelease, String irBuildMetadata, long opsetVersion,
			String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

	@Override
	public SumV8<Tensor<?>> getSumV8() {
		// TODO Auto-generated method stub
		return null;
	}

}