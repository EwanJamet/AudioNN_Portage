import onnx

model_path = "/home/ewan/Desktop/audioset_tagging_cnn/MobileNetTens/MobileNetTens_W.onnx"
onnx_model = onnx.load(model_path)
print(onnx.checker.check_model(onnx_model))
