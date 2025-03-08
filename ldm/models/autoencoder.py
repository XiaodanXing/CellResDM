import pandas as pd
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

class VQModelTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, force_not_quantize=True)
        return dec

    def forward(self, input, force_not_quantize=False):
        h = self.encode(input)
        dec = self.decode(h, force_not_quantize)
        return dec

class AutoencoderKLTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input, sample_posterior, return_moments=False)
        dec = self.decode(z)
        return dec

class EncoderKLTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z
    def forward(self, x, sample_posterior=True, return_moments=False):
        return self.encode(x, sample_posterior, return_moments)

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

if __name__ == '__main__':
    import torch
    from torchvision import transforms
    from PIL import Image
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf


    def load_image(image_path, size=512):
        """Load and transform an image."""
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
        ])
        images = []
        for i in range(1,6):
            image = Image.open(image_path.replace('ch4','ch%i'%i))
            image = transform(image)
            images.append(image)
        images = torch.cat(images)
        # image = Image.open(image_path).convert('RGB')
        # image = transform(image).unsqueeze(0)  # Add batch dimension
        # image = torch.concat([image, image], dim=1)
        # image = image[:,:5,...]
        images = images.unsqueeze(0)
        return images


    def test_model(model, image_path):
        """Test the VQ autoencoder model on a single image."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Load and preprocess the image
        image = load_image(image_path).to(device)

        # Forward pass through the model
        with torch.no_grad():
            reconstructed_image = model(image)

        # Convert the tensor to an image and display it
        # output_img = transforms.ToPILImage()(reconstructed_image.squeeze(0).cpu())
        # input_img = transforms.ToPILImage()(image.squeeze(0).cpu())
        reconstructed_image = reconstructed_image.squeeze(0).cpu()
        image = image.squeeze(0).cpu()
        output_img = torch.cat([reconstructed_image[i] for i in range(reconstructed_image.size(0))], dim=1)
        input_img = torch.cat([image[i] for i in range(image.size(0))], dim=1)

        plt.subplot(1,2,1)
        plt.imshow(output_img,cmap='Greys_r')
        plt.title('Reconstructed Image')
        plt.subplot(1,2,2)
        plt.imshow(input_img,cmap='Greys_r')
        plt.title('Original Image')
        plt.show()


    configs = OmegaConf.load('/home/xiaodan/PycharmProjects/'
                             'cell_painting_diffusion/ResShift/configs/custom.yaml')
    # ckpt = torch.load('/home/xiaodan/PycharmProjects/'
    #                   'cell_painting_diffusion/ResShift/'
    #                   'weights/autoencoder_vq_f4.pth', map_location=f"cuda:0")
    ckpt_path = '/home/xiaodan/PycharmProjects/IPMI2023/diffusion_model/' \
                'logs/2024-05-29T09-55-06_autoencoder_vq_f4/checkpoints/last.ckpt'
    # ckpt_path = '/home/xiaodan/PycharmProjects/IPMI2023/diffusion_model/' \
    #             'logs/2024-05-27T11-13-33_autoencoder_vq_f4/checkpoints/epoch=000008.ckpt'
    ckpt = torch.load(ckpt_path, map_location=f"cuda:0")['state_dict']
    # ckpt = torch.load('/home/xiaodan/Downloads/vq-f4/model.ckpt', map_location=f"cuda:0")['state_dict']
    keys_to_remove = [key for key in ckpt.keys() if 'loss' in key]
    for key in keys_to_remove:
        del ckpt[key]
    model = VQModelTorch(**configs.autoencoder.params)
    model.load_state_dict(ckpt)
    # Assuming `model` is an instance of `VQModelTorch` that has been trained
    test_model(model, '/media/xiaodan/cellpainting/predictions/'
                      'real/smallval/Images/r01c01f03p01-ch4sk1fk1fl1.tiff')
    # test_model(model, '/home/xiaodan/Downloads/images.jpeg')
