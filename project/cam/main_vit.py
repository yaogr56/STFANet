import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from cam.utils import GradCAM, show_cam_on_image, center_crop_img
from model.STnet_lastthreebranch import STFusion
# from vit_model import vit_base_patch16_224


class ReshapeTransform:
    def __init__(self, model):
        # input_size = model.patch_embed.img_size
        # patch_size = model.patch_embed.patch_size
        # self.h = input_size[0] // patch_size[0]
        # self.w = input_size[1] // patch_size[1]
        self.h = 1
        self.w = 1
    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    model = STFusion()
    # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    weights_path = r"D:\pythonProject1\trainedmodel\STFullbranch-dfdc_epoch-239.pth.tar"
    # model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(0))
    checkpoint = torch.load(weights_path, map_location="cpu")
    print("Initializing weights form: {}...".format(weights_path))
    model.load_state_dict(checkpoint["state_dict"])
    # Since the final classification is done on the class token computed
    # in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    target_layers = [model.vit.transformer.layers[0][1].fn.norm]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = r"D:\pythonProject1\000012.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.savefig('cam_dfdc.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()