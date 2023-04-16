from thop import profile
import torch
from torchvision.models import resnet50
import time

def cal_eff_score(count = 100, use_cuda=True):

    # define input tensor
    inp_tensor = torch.rand(1, 3, 1080, 1920) # NOTE: this is the shape for ACDC images

    # define model
    model = resnet50()

    # deploy to cuda
    if use_cuda:
        inp_tensor = inp_tensor.cuda()
        model = model.cuda()

    # get flops and params
    flops, params = profile(model, inputs=(inp_tensor, ))
    G_flops = flops * 1e-9
    M_params = params * 1e-6

    # get time
    start_time = time.time()
    for i in range(count):
        _ = model(inp_tensor)
    used_time = time.time() - start_time
    ave_time = used_time / count

    # print score
    print('FLOPs (G) = {:.4f}'.format(G_flops))
    print('Params (M) = {:.4f}'.format(M_params))
    print('Time (S) = {:.4f}'.format(ave_time))

if __name__ == "__main__":
    cal_eff_score()
