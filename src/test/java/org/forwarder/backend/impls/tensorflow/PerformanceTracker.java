package org.forwarder.backend.impls.tensorflow;

public class PerformanceTracker {
	
	private long startTime;
	
	public static PerformanceTracker start() {
		return new PerformanceTracker();
	}
	
	public long stop() {
		return System.currentTimeMillis() - this.startTime;
	}
	
	private PerformanceTracker() {
		this.startTime = System.currentTimeMillis();
	}

}
