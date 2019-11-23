import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

# 可以根据模块图形的数值设置输入输出的显示名称。这些设置不会改变此图形的语义。只是会变得更加可读了。
#该网络的输入包含了输入的扁平表(flat list)。也就是说传入forward()里面的值，其后是扁平表的参数。你可以指定一部分名字，例如指定一个比该模块输入数量更少的表，随后我们会从一开始就设定名字。
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)