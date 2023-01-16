import torch
from model import ToMModel
from dataset import ToMDataset

dataset = ToMDataset('/home/yilue/datasets/pick_best_400', max_data_num=1)
model = ToMModel(dataconfig=dataset.config)
model.load_state_dict(torch.load('tom_ckpt/tom_5.pth'))

d_input = dataset[0][0]
d_input_1, d_input_2 = d_input
d_input_1 = d_input_1.unsqueeze(0).cuda()
d_input_2 = d_input_2.unsqueeze(0).cuda()
d_input = [d_input_1, d_input_2]
# d_output = model(d_input)
# d_output = model((d_input_1,d_input_2))

torch.onnx.export(model.cuda(), d_input, 'tomcuda.onnx', verbose=True,
                  input_names=['act_seq', 'obs'], output_names=['output'])
