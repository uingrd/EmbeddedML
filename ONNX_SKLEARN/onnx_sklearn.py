#!/usr/bin/python3.5
# coding=utf-8

import numpy as np

np.random.seed(1234)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

########################################
# 演示：将scikit-learn模型生成ONNX格式模型文件
# 这一模型需要用ONNXruntime在PC上运行
# 代码清单 8-25
########################################

## 数据加载
print('[INF] loading data...')
iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y)

## 模型训练
print('[INF] model trainning...')
model = RandomForestClassifier()

## 训练结果验证
print('[INF] verifying...')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('[INF] y_pred:\n    ',y_pred)
print('[INF] Number of mislabeled points out of a total %d points : %d'%
      (len(y_test),(y_test != y_pred).sum()))

## 生成ONNX格式数据文件
print('[INF] generating ONNX data...')
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64Type
from skl2onnx.helpers import onnx_helper

initial_type = [('float_input', FloatTensorType([None, 4]))]
model_onnx = convert_sklearn(model, initial_types=initial_type)

with open('model.onnx', 'wb') as f:
    f.write(model_onnx.SerializeToString())

## 加载ONNX模型，并运行
if True:
    print('[INF] loading ONNX model...')
    import onnxruntime as rt
    import numpy
    sess = rt.InferenceSession('model.onnx')
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    print('[INF] name of input node:',input_name)
    print('[INF] name of output node:',label_name)
    y_pred_onx = sess.run([label_name], {input_name: x_test.astype(numpy.float32)})[0]
    print('[INF] y_pred_onx:\n    ',y_pred_onx)
    print('[INF] Number of mislabeled points out of a total %d points : %d'%
          (len(y_test),(y_test != y_pred_onx).sum()))


