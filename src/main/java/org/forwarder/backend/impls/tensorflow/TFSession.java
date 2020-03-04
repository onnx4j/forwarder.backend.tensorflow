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
package org.forwarder.backend.impls.tensorflow;

import org.forwarder.Session;
import org.tensorflow.EagerSession;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;

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

	public static Scope getScope() {
		return TFSession.getSession().tfScope;
	}
	
	public static TFOps getOps() {
		return TFSession.getSession().tfOps;
	}

	private EagerSession tfSession;
	private Scope tfScope;
	private TFOps tfOps;

	public TFSession(TFBackend backend) {
		super(backend);

		this.tfSession = this.createTFSession();
		this.tfScope = new Scope(this.tfSession);
		this.tfOps = new TFOps(this.tfScope);
	}
	
	@Override
	public void close() throws Exception {
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