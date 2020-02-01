# forwarder.backend.tensorflow
## 简介
A backend for Forwarder using Google Tensorflow.

## Operator支持
### ai.onnx Operators
|Operator|Opset1|Opset2|Opset3|Opset4|Opset5|Opset6|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
|Abs|1|1|1|1|1|1|
|Add|1|1|1|1|1|1|
|ArgMax|1|1|1|1|1|1|
|AveragePool|1|1|1|1|1|1|
|BatchNormalization|1|1|1|1|1|1|
|Concat|1|1|1|4|4|4|
|Constant|1|1|1|1|1|1|
|Conv|1|1|1|1|1|1|
|Div|1|1|1|1|1|1|
|Dropout|1|1|1|1|1|6|
|Gather|1|1|1|1|1|1|
|Identity|1|1|1|1|1|1|
|ImageScaler|1|1|1|1|1|1|
|LeakyRelu|1|1|1|1|1|1|
|MatMul|1|1|1|1|1|1|
|MaxPool|1|1|1|1|1|1|
|Mul|1|1|1|1|1|6|
|Pad|1|1|1|1|1|1|
|Relu|1|1|1|1|1|1|
|Reshape|1|1|1|1|5|5|
|Shape|1|1|1|1|1|1|
|Transpose|1|1|1|1|1|1|

### ai.onnx.ml Operators
暂不支持。
