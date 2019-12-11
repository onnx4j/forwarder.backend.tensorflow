package org.forwarder.backend.impls.tensorflow;

import org.forwarder.Session;
import org.forwarder.executor.Executor;
import org.tensorflow.EagerSession;
import org.tensorflow.Tensor;

public class TFSession extends Session<Tensor<?>> {

	public static TFSession getSession() {
		Session<?> sess = TL_SESSION.get();
		if (sess == null)
			throw new RuntimeException(
					"No forwarder session binded in this thread");

		if (TFSession.class.isInstance(sess) == false)
			throw new java.lang.ClassCastException(
					"Session class type not match");

		TFSession tfSess = TFSession.class.cast(sess);
		return tfSess;
	}

	public static EagerSession get() {
		return TFSession.getSession().tfSession;
	}

	private EagerSession tfSession;

	public TFSession(Executor<Tensor<?>> executor, TFBackend backend) {
		super(executor, backend);

		this.tfSession = this.createTFSession();
	}
	
	@Override
	public void close() {
		super.close();
		this.disposeTFSession();
	}

	/**
	 * 创建后端实现Session实例
	 * 
	 * @return
	 */
	protected EagerSession createTFSession() {
		return EagerSession.create();
	}

	/**
	 * 释放后端实现Session资源
	 */
	protected void disposeTFSession() {
		if (this.tfSession != null)
			this.tfSession.close();
	}

}
