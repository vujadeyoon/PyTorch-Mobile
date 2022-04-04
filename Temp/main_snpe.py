import torch
from demo_superpoint import SuperPointNet, SuperPointFrontend


if __name__=='__main__':
    weights_path = 'superpoint_v1.pth'
    nms_dist = 4
    conf_thresh = 0.015
    nn_thresh = 0.7
    cuda = False

    net = SuperPointNet()
    if cuda:
        # Train on GPU, deploy on GPU.
        net.load_state_dict(torch.load(weights_path))
        net = net.to('cuda:0')
    else:
        # Train on GPU, deploy on CPU.
        net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))

    net.eval()

    input_shape = [1, 1, 120, 160]
    input_data = torch.randn(input_shape)
    script_model = torch.jit.trace(net, input_data)
    script_model.save("./fe.pt")
