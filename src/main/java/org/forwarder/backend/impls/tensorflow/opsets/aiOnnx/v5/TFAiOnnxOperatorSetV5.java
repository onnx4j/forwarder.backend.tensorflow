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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v5;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v4.TFAiOnnxOperatorSetV4;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v5.ops.TFReshapeV5;
import org.onnx4j.opsets.domain.aiOnnx.v5.AiOnnxOpsetInitializerV5;
import org.onnx4j.opsets.domain.aiOnnx.v5.ops.ReshapeV5;

public class TFAiOnnxOperatorSetV5 extends TFAiOnnxOperatorSetV4 implements AiOnnxOpsetInitializerV5 {

	@Override
	public ReshapeV5 getReshapeV5() { return new TFReshapeV5(); }

	public TFAiOnnxOperatorSetV5() {
		super(1, "", "", 5L, "ONNX OPSET-V5 USING TENSORFLOW BACKEND");
	}

	public TFAiOnnxOperatorSetV5(int irVersion, String irVersionPrerelease, String irBuildMetadata,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

}