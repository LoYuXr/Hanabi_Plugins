import onnx
from onnx_tf.backend import prepare

# model_onnx = onnx.load('tomcuda.onnx')
# tf_rep = prepare(model_onnx)
# # Export model as .pb file
# tf_rep.export_graph('./model/tomcuda.pb')


def onnx2pb(onnx_input_path, pb_output_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model

if __name__ == "__main__":
    onnx_input_path = 'tomcuda.onnx'
    pb_output_path = 'tomcuda.pb'
    onnx2pb(onnx_input_path, pb_output_path)

print("done")