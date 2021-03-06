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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v9;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v8.TFAiOnnxOperatorSetV8;
import org.onnx4j.opsets.domain.aiOnnx.v9.AiOnnxOpsetInitializerV9;
import org.onnx4j.opsets.domain.aiOnnx.v9.ops.CastV9;

public class TFAiOnnxOperatorSetV9 extends TFAiOnnxOperatorSetV8 implements AiOnnxOpsetInitializerV9 {

	public TFAiOnnxOperatorSetV9() {
		super(1, "", "", 9L, "ONNX OPSET-V9 USING TENSORFLOW BACKEND");
	}

	public TFAiOnnxOperatorSetV9(int irVersion, String irVersionPrerelease, String irBuildMetadata, long opsetVersion,
			String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

	@Override
	public CastV9 getCastV9() {
		// TODO Auto-generated method stub
		return null;
	}

}