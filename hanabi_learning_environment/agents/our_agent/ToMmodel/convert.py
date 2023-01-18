import torch
from model import ToMModel
from dataset import ToMDataset, Config

config = Config()
config.num_players = 3

dataset = ToMDataset('/home/yilue/datasets/files', max_data_num=1, config=config)
model = ToMModel(dataconfig=dataset.config)
model.load_state_dict(torch.load('tom_ckpt3/tom_16.pth'))

d_input = dataset[0][0]
d_input_1, d_input_2 = d_input
d_input_1 = d_input_1.unsqueeze(0).to('cuda')
d_input_2 = d_input_2.unsqueeze(0).to('cuda')
d_input = [d_input_1, d_input_2]
# d_output = model(d_input)
# d_output = model((d_input_1,d_input_2))

torch.onnx.export(model.cuda(), d_input, 'tom3.onnx', verbose=True,
                  input_names=['act_seq', 'obs'], output_names=['output'])
