import os
import torch
from model.VideoTransformer import SwinTransformer3D

pretrained = r'D:\pythonProject\VideoSwinTransformer\tools\swin_tiny_patch244_window877_kinetics400_1k.pth'
net = SwinTransformer3D()
print(net)
pre_weights_dict = torch.load(pretrained, map_location='cuda:0')['state_dict']

# print(model)
del_key = []
for k, _ in pre_weights_dict.items():
    if 'head' in k:
        del_key.append(k)
for k in del_key:
    del pre_weights_dict[k]

values = list(pre_weights_dict.values())
new_keys = []
print('-----------------------')

for k, _ in pre_weights_dict.items():
    k = k.split('.', 1)[1:]
    new_keys = new_keys + k
print(new_keys)

pre_weights_dict = dict(zip(new_keys, values))

# print('--------------------------')
# print(pre_weights_dict.keys())

# for k, _ in pre_weights_dict.items():
#     k = k.split('.', 1)[1:]
#     new_keys.append(k)
#
# print(new_keys)

# pre_weights_dict = dict(zip(new_keys, values))
# print(type(values))
# print(type(new_keys))
# print(pre_weights_dict)
# for k, _ in pre_weights_dict.items():
#
#
#     pre_weights_dict.update({nk[0]: pre_weights_dict.pop(k)})
#     # print(k)
#     # print(nk)

# print("*****", pre_weights_dict.keys())
misskeys, unexpected_keys = net.load_state_dict(pre_weights_dict, strict=False)
print("misskeys:", misskeys)
print(unexpected_keys)
print('-------------------')
print(net.parameters())
print(list(net.parameters()))

print('______________________________')
# # print(del_key)

# x = torch.Tensor(1,3,32,224,224)
# # print(x)
# # print(x.shape)
# output = model(x)
# print(output)
# print(output.shape)