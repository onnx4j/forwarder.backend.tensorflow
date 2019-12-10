package org.forwarder.backend.impls.tensorflow.opsets;

import org.forwarder.backend.impls.tensorflow.TFSession;

import junit.framework.TestCase;

/**
 * Unit test for some performance tests.
 */
public class TFOperatorTest extends TestCase {
	
	private TFSession session;

	public TFOperatorTest(String testName) {
		super(testName);
		
		this.session = new TFSession(null);
	}
	
	public TFSession getSession() {
		return this.session;
	}

}
