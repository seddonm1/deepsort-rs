import onnx
import onnx_graphsurgeon as gs
import numpy as np

filename = "../mars-small128.onnx"

model = onnx.load(filename)
graph = gs.import_onnx(model)

# change input type
graph.inputs[0].dtype=np.float32

# replace input
conv = [node for node in graph.nodes if node.name == "conv1_1/Conv2D__40"][0]
conv.inputs[0] = graph.inputs[0]

# prepare export
graph.fold_constants()
graph.cleanup().toposort()
model = gs.export_onnx(graph)

onnx.save(model, filename)