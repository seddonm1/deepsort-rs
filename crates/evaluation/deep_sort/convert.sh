# pip3 install tensorflow-cpu tf2onnx nvidia-pyindex
# pip3 install onnx-graphsurgeon
python3 -m tf2onnx.convert --input mars-small128.pb --inputs images:0 --outputs features:0 --output ../mars-small128.onnx --verbose --rename-inputs images --rename-outputs features --opset 14
