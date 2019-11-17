import taso
import onnx

#Build DNN model
graph = taso.new_graph()
input = graph.new_input(dims=(1,128,56,56))
w1 = graph.new_weight(dims=(128,128,3,3))
w2 = graph.new_weight(dims=(128,128,1,1))
w3 = graph.new_weight(dims=(128,128,3,3))
left = graph.conv2d(input=input, weight=w1, strides=(1,1), padding="SAME", activation="RELU")
left = graph.conv2d(input=input, weight=w3, strides=(1,1), padding="SAME")
right = graph.conv2d(input=input, weight=w2, strides=(1,1), padding="SAME", activation="RELU")
output = graph.add(left, right)
output = graph.relu(output)

#Optimize DNN model
new_graph = taso.optimize(graph)
onnx_model = taso.export_onnx(new_graph)
onnx.save(onnx_model, "arbitrary_DNN.onnx")