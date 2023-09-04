
import math

import torch
from torch import nn
from torch.nn import init


class ViewSpecificVAE(nn.Module):
    
    def __init__(self, args, channels=1) -> None:
        super().__init__()
        self.args = args
        
        self.img_shape = args.valid_augmentation.crop_size
        

        # view-specific id.
        self.num_classes = self.args.dataset.class_num
        self.latent_dim = self.args.vspecific.latent_dim
        
        self.expands = self.args.vspecific.expands
        
        self.hidden_dims = self.args.vspecific.hidden_dims
        
        
        self.input_channel = channels
        self.build_encoder_and_decoder()
        
        
        self.recons_criterion = nn.BCELoss()
        self.apply(self.weights_init(init_type=self.args.backbone.init_method))
        
        
    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0.0)

        return init_fun
                
        
    def build_encoder_and_decoder(self):
        if self.img_shape == 64:
            self.stem = nn.Sequential(
                nn.Conv2d(self.input_channel, 32, 4, 2, 1),          # B,  32, 32, 32
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
                nn.ReLU(True),
            )
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose2d(32, self.input_channel, 4, 2, 1),  # B, nc, 64, 64
                # nn.Tanh(),
                nn.Sigmoid(),
            )
        elif self.img_shape == 32:
            self.stem = nn.Sequential(
                nn.Conv2d(self.input_channel, 32, 4, 2, 1),          # B,  32, 32, 32
                nn.ReLU(True),
            )
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(32, self.input_channel, 4, 2, 1),  # B, nc, 32, 32
                nn.Sigmoid(),
                # nn.Tanh(),
            )
        else:
            raise ValueError("img shape must in [32, 64]")
        self._encoder = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  128,  4,  4
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            
        )
        self.latent2dist = nn.Linear(256, self.latent_dim*2)
        self.dist2latent = nn.Linear(self.latent_dim+self.num_classes, 256)
        # self.dist2latent = nn.Linear(self.latent_dim+1, 256)
        self._decoder = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4),      # B,  128,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True)
        )
        
    
    
    def latent(self, x, mask_idx=None):
        x = self.stem(x)
        latent = self._encoder(x)
        latent = torch.flatten(latent, start_dim=1)
        return latent
        
    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        x = self.stem(x)
        latent = self._encoder(x)
        latent = torch.flatten(latent, start_dim=1)
        distributions = self.latent2dist(latent)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = distributions[:, :self.latent_dim]
        logvar = distributions[:, self.latent_dim:]

        return [mu, logvar]

    def decode(self, z):
        # if self.expands > 2:
        #     pixel = self.expands // 2
        # else:
        #     pixel = self.expands
        result = self.dist2latent(z)
        result = result.view(-1, 256, 1, 1)
        result = self._decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, x, y, mask_idx=None):
        
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, y], dim=1)
            
        return self.decode(z), mu, logvar
    
    
    def get_loss(self, x, y=None, mask_idx: torch.BoolTensor = None):
        out, mu, logvar = self(x, y)
        
        if mask_idx is not None:
            # ignore missing instances
            out = out[mask_idx]
            x = x[mask_idx]
            mu = mu[mask_idx]
            logvar = logvar[mask_idx]
        
        recons_loss = self.recons_criterion(out, x)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        
        return recons_loss, kld_loss
    
    
    def sample(self, num_samples, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param y: (Tensor) controlled labels.
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(self.device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples