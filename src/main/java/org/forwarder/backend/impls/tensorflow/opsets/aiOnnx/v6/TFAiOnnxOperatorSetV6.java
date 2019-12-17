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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v6;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v5.TFAiOnnxOperatorSetV5;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v6.ops.TFDropoutV6;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v6.ops.TFMulV6;
import org.onnx4j.opsets.aiOnnx.v6.AiOnnxOperatorSetSpecV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.CastV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.DropoutV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.MulV6;
import org.tensorflow.Tensor;

public class TFAiOnnxOperatorSetV6 extends TFAiOnnxOperatorSetV5 implements AiOnnxOperatorSetSpecV6<Tensor<?>> {

	@Override
	public MulV6<Tensor<?>> getMulV6() {
		return new TFMulV6();
	}

	@Override
	public DropoutV6<Tensor<?>> getDropoutV6() {
		return new TFDropoutV6();
	}

	@Override
	public CastV6<Tensor<?>> getCastV6() { return null; }

	public TFAiOnnxOperatorSetV6() {
		super(1, "", "", 6L, "ONNX OPSET-V6 USING TENSORFLOW BACKEND");
	}

	public TFAiOnnxOperatorSetV6(int irVersion, String irVersionPrerelease, String irBuildMetadata, long opsetVersion,
			String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

}