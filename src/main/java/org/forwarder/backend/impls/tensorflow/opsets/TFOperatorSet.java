package org.forwarder.backend.impls.tensorflow.opsets;

import org.forwarder.backend.impls.tensorflow.TFBackend;
import org.forwarder.opset.annotations.Opset;
import org.onnx4j.opsets.OperatorSet;

@Opset(backendName = TFBackend.BACKEND_NAME)
public abstract class TFOperatorSet extends OperatorSet {

	public TFOperatorSet(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
		// TODO Auto-generated constructor stub
	}

}
