package org.forwarder.backend.impls.tensorflow;

import org.apache.commons.lang3.Range;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;

public final class TFOps {

	private Scope scope;
	private Ops ops;

	public TFOps(Scope scope) {
		this.scope = scope;
		this.ops = Ops.create(this.scope.env());
	}
	
	public Scope scope() {
		return this.scope;
	}

	public Ops ops() {
		return this.ops;
	}

	public <T extends Number> Operand<T> broadcast(TFOps tfOps, Tensor<T> a, Tensor<T> b, Long axis) {
		Operand<T> operandB = tfOps.constant(b);
		
		if (axis == null)
			return operandB;

		if (axis + b.numDimensions() == a.numDimensions())
			return operandB;

		if (axis < 0L)
			axis += a.numDimensions();

		Range<Integer> keepdims = Range.between(axis.intValue(), axis.intValue() + b.numDimensions() - 1);
		int nbDiffDims = a.numDimensions() - b.numDimensions();
		for (int n = 0; n <= nbDiffDims; n++) {
			if (keepdims.contains(n) == false) {
				Constant<Long> opAxis = tfOps.ops().constant(Long.valueOf(n));
				operandB = tfOps.ops().expandDims(operandB, opAxis).asOutput();
			}
		}
		return operandB;
	}

	public Operand<Number> toNCHW(Operand<Number> inputNHWC) {
		return ops.linalg.transpose(inputNHWC, ops.constant(new int[] { 0, 3, 1, 2 }));
	}

	public Operand<Number> toNHWC(Operand<Number> inputNCHW) {
		return ops.linalg.transpose(inputNCHW, ops.constant(new int[] { 0, 2, 3, 1 }));
	}

	public Operand<Number> toHWCN(Operand<Number> inputNCHW) {
		return ops.linalg.transpose(inputNCHW, Constant.create(scope, new int[] { 2, 3, 1, 0 }));
	}

	public <T> Output<T> constant(Tensor<T> x) {
		OperationBuilder opBuilder = this.prepare("Const");
		opBuilder.setAttr("dtype", x.dataType());
		opBuilder.setAttr("value", x);
		return this.build(opBuilder).output(0);
	}

	private OperationBuilder prepare(String opName) {
		return scope.env().opBuilder(opName, scope.makeOpName(opName));
	}

	private Operation build(OperationBuilder opBuilder) {
		return opBuilder.build();
	}

}
