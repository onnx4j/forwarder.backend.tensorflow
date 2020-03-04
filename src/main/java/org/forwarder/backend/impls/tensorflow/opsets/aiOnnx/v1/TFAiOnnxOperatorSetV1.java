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
package org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1;

import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.TFAiOnnxOperatorSet;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFAbsV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFAddV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFArgMaxV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFAveragePoolV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFBatchNormalizationV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFConcatV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFConstantV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFConvV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFDivV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFDropoutV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFIdentityV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFImageScalerV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFLeakyReluV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFMatMulV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFMaxPoolV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFMulV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFReluV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFReshapeV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFShapeV1;
import org.forwarder.backend.impls.tensorflow.opsets.aiOnnx.v1.ops.TFTransposeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOpsetInitializerV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AbsV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AddV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ArgMaxV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AveragePoolV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.BatchNormalizationV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.CastV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConcatV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConstantV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConvV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.DivV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.DropoutV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.GatherV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.IdentityV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ImageScalerV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.LeakyReluV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MatMulV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MaxPoolV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MulV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReduceMaxV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReluV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReshapeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ShapeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SigmoidV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SoftmaxV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SqueezeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SubV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SumV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.TransposeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.UnsqueezeV1;

public class TFAiOnnxOperatorSetV1 extends TFAiOnnxOperatorSet implements AiOnnxOpsetInitializerV1 {

	@Override
	public AbsV1 getAbsV1() { return new TFAbsV1(); }

	//@Override
	//public PadV1 getPadV1() { return new TFPadV1(); }

	@Override
	public MatMulV1 getMatMulV1() { return new TFMatMulV1(); }

	@Override
	public IdentityV1 getIdentityV1() { return new TFIdentityV1(); }

	@Override
	public ArgMaxV1 getArgMaxV1() { return new TFArgMaxV1(); }

	@Override
	public DivV1 getDivV1() { return new TFDivV1(); }

	@Override
	public ReshapeV1 getReshapeV1() { return new TFReshapeV1(); }

	@Override
	public MaxPoolV1 getMaxPoolV1() { return new TFMaxPoolV1(); }

	@Override
	public AddV1 getAddV1() { return new TFAddV1(); }

	@Override
	public ConstantV1 getConstantV1() { return new TFConstantV1(); }

	@Override
	public ReluV1 getReluV1() { return new TFReluV1(); }

	@Override
	public ConvV1 getConvV1() { return new TFConvV1(); }

	@Override
	public ImageScalerV1 getImageScalerV1() { return new TFImageScalerV1(); }

	@Override
	public BatchNormalizationV1 getBatchNormalizationV1() { return new TFBatchNormalizationV1(); }

	@Override
	public LeakyReluV1 getLeakyReluV1() { return new TFLeakyReluV1(); }

	@Override
	public MulV1 getMulV1() { return new TFMulV1(); }

	@Override
	public ConcatV1 getConcatV1() { return new TFConcatV1(); }

	@Override
	public DropoutV1 getDropoutV1() { return new TFDropoutV1(); }

	@Override
	public AveragePoolV1 getAveragePoolV1() { return new TFAveragePoolV1(); }

	@Override
	public CastV1 getCastV1() { return null; }

	@Override
	public GatherV1 getGatherV1() { return null; }

	public TFAiOnnxOperatorSetV1() {
		this(1, "", "", 1L, "ONNX OPSET-V1 USING TENSORFLOW BACKEND");
	}

	public TFAiOnnxOperatorSetV1(int irVersion, String irVersionPrerelease, String irBuildMetadata,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

	@Override
	public SubV1 getSubV1() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SumV1 getSumV1() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SigmoidV1 getSigmoidV1() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SoftmaxV1 getSoftmaxV1() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SqueezeV1 getSqueezeV1() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public UnsqueezeV1 getUnsqueezeV1() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ReduceMaxV1 getReduceMaxV1() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TransposeV1 getTransposeV1() { return new TFTransposeV1(); }

	@Override
	public ShapeV1 getShapeV1() { return new TFShapeV1(); }

}