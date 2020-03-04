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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v7;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v6.TFAiOnnxOperatorSetV6;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v7.ops.TFAveragePoolV7;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v7.ops.TFBatchNormalizationV7;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v7.ops.TFDropoutV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.AiOnnxOperatorSetInitializerV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.AveragePoolV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.BatchNormalizationV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.DropoutV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.SubV7;

public class TFAiOnnxOperatorSetV7 extends TFAiOnnxOperatorSetV6 implements AiOnnxOperatorSetInitializerV7 {

	public TFAiOnnxOperatorSetV7() {
		super(1, "", "", 7L, "ONNX OPSET-V7 USING TENSORFLOW BACKEND");
	}

	public TFAiOnnxOperatorSetV7(int irVersion, String irVersionPrerelease, String irBuildMetadata, long opsetVersion,
			String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

	@Override
	public BatchNormalizationV7 getBatchNormalizationV7() {
		return new TFBatchNormalizationV7();
	}

	@Override
	public AveragePoolV7 getAveragePoolV7() {
		return new TFAveragePoolV7();
	}

	@Override
	public DropoutV7 getDropoutV7() {
		return new TFDropoutV7();
	}

	@Override
	public SubV7 getSubV7() {
		// TODO Auto-generated method stub
		return null;
	}

}