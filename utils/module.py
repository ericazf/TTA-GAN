import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn import Parameter
import math 

class PrototypeNet(nn.Module):
    def __init__(self, bit, num_classes):
        super(PrototypeNet, self).__init__()

        self.feature = nn.Sequential(nn.Linear(num_classes, 4096),
                                     nn.ReLU(True), nn.Linear(4096, 512))
        self.hashing = nn.Sequential(nn.Linear(512, bit), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(512, num_classes),
                                          nn.Sigmoid())

    def forward(self, label):
        f = self.feature(label)
        h = self.hashing(f)
        c = self.classifier(f)
        return f, h, c

#--------------------------TTA-GAN--------------------------------------
class GeneratorCls(nn.Module):
    """Generator: Encoder-Decoder Architecture.
    Reference: https://github.com/yunjey/stargan/blob/master/model.py
    """
    def __init__(self, num_classes):
        super(GeneratorCls, self).__init__()

        # Label Encoder
        self.label_encoder = LabelEncoderCls(num_classes)

        # Image Encoder
        curr_dim = 64
        image_encoder = [
            nn.Conv2d(6, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True)
        ]
        # Down Sampling
        for i in range(2):
            image_encoder += [
                nn.Conv2d(curr_dim,
                          curr_dim * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=True),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim * 2
        # Bottleneck
        for i in range(3):
            image_encoder += [
                ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')
            ]
        self.image_encoder = nn.Sequential(*image_encoder)

        # Decoder
        decoder = []
        # Bottleneck
        for i in range(3):
            decoder += [
                ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')
            ]
        # Up Sampling
        for i in range(2):
            decoder += [
                nn.ConvTranspose2d(curr_dim,
                                   curr_dim // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2
        self.residual = nn.Sequential(
            # nn.Conv2d(curr_dim + 3,
            #           curr_dim,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1,
            #           bias=False),
            # nn.InstanceNorm2d(curr_dim // 2, affine=False),
            # nn.ReLU(inplace=True),
            nn.Conv2d(curr_dim + 3,
                      3,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.Tanh())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, label_feature):
        mixed_feature = self.label_encoder(x, label_feature)
        encode = self.image_encoder(mixed_feature)
        decode = self.decoder(encode)
        decode_x = torch.cat([decode, x], dim=1)
        adv_x = self.residual(decode_x)
        return adv_x, mixed_feature

class LabelEncoderCls(nn.Module):
    def __init__(self, num_classes, nf=128):
        super(LabelEncoderCls, self).__init__()
        self.nf = nf
        curr_dim = nf
        self.size = 14

        self.fc = nn.Sequential(
            # nn.Linear(512, 512), nn.ReLU(True), 
            nn.Linear(num_classes, curr_dim * self.size * self.size), nn.ReLU(True))

        transform = []
        for i in range(4):
            transform += [
                nn.ConvTranspose2d(curr_dim,
                                   curr_dim // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                # nn.Upsample(scale_factor=(2, 2)),
                # nn.Conv2d(curr_dim, curr_dim//2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=False),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2

        transform += [
            nn.Conv2d(curr_dim,
                      3,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False)
        ]
        self.transform = nn.Sequential(*transform)

    def forward(self, image, label_feature):
        label_feature = self.fc(label_feature)
        label_feature = label_feature.view(label_feature.size(0), self.nf, self.size, self.size)
        label_feature = self.transform(label_feature)

        # mixed_feature = label_feature + image
        mixed_feature = torch.cat((label_feature, image), dim=1)
        return mixed_feature
        

class MyDiscriminator(nn.Module):
    """
    Discriminator network with PatchGAN.
    Reference: https://github.com/yunjey/stargan/blob/master/model.py
    """
    def __init__(self, num_classes, image_size=224, conv_dim=64, repeat_num=5):
        super(MyDiscriminator, self).__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / (2**repeat_num))
        self.main = nn.Sequential(*layers)
        # self.fc = nn.Conv2d(curr_dim, num_classes + 1, kernel_size=kernel_size, bias=False)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size = kernel_size, bias = False)
        self.conv2 = nn.Conv2d(curr_dim, num_classes, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src.squeeze().unsqueeze(1), out_cls.squeeze()


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine))

    def forward(self, x):
        return x + self.main(x)




# #-----------------------------pros_gan--------------------------------
# class Generator(nn.Module):
#     """Generator: Encoder-Decoder Architecture.
#     Reference: https://github.com/yunjey/stargan/blob/master/model.py
#     """
#     def __init__(self):
#         super(Generator, self).__init__()

#         # Label Encoder
#         self.label_encoder = LabelEncoder()

#         # Image Encoder
#         curr_dim = 64
#         image_encoder = [
#             nn.Conv2d(6, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
#             nn.InstanceNorm2d(curr_dim),
#             nn.ReLU(inplace=True)
#         ]
#         # Down Sampling
#         for i in range(2):
#             image_encoder += [
#                 nn.Conv2d(curr_dim,
#                           curr_dim * 2,
#                           kernel_size=4,
#                           stride=2,
#                           padding=1,
#                           bias=True),
#                 nn.InstanceNorm2d(curr_dim * 2),
#                 nn.ReLU(inplace=True)
#             ]
#             curr_dim = curr_dim * 2
#         # Bottleneck
#         for i in range(3):
#             image_encoder += [
#                 ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')
#             ]
#         self.image_encoder = nn.Sequential(*image_encoder)

#         # Decoder
#         decoder = []
#         # Bottleneck
#         for i in range(3):
#             decoder += [
#                 ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')
#             ]
#         # Up Sampling
#         for i in range(2):
#             decoder += [
#                 nn.ConvTranspose2d(curr_dim,
#                                    curr_dim // 2,
#                                    kernel_size=4,
#                                    stride=2,
#                                    padding=1,
#                                    bias=False),
#                 nn.InstanceNorm2d(curr_dim // 2),
#                 nn.ReLU(inplace=True)
#             ]
#             curr_dim = curr_dim // 2
#         self.residual = nn.Sequential(
#             # nn.Conv2d(curr_dim + 3,
#             #           curr_dim,
#             #           kernel_size=3,
#             #           stride=1,
#             #           padding=1,
#             #           bias=False),
#             # nn.InstanceNorm2d(curr_dim // 2, affine=False),
#             # nn.ReLU(inplace=True),
#             nn.Conv2d(curr_dim + 3,
#                       3,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1,
#                       bias=False), nn.Tanh())
#         self.decoder = nn.Sequential(*decoder)

#     def forward(self, x, label_feature):
#         mixed_feature = self.label_encoder(x, label_feature)
#         encode = self.image_encoder(mixed_feature)
#         decode = self.decoder(encode)
#         decode_x = torch.cat([decode, x], dim=1)
#         adv_x = self.residual(decode_x)
#         return adv_x, mixed_feature

# class LabelEncoder(nn.Module):
#     def __init__(self, nf=128):
#         super(LabelEncoder, self).__init__()
#         self.nf = nf
#         curr_dim = nf
#         self.size = 14

#         self.fc = nn.Sequential(
#             # nn.Linear(512, 512), nn.ReLU(True), 
#             nn.Linear(512, curr_dim * self.size * self.size), nn.ReLU(True))

#         transform = []
#         for i in range(4):
#             transform += [
#                 nn.ConvTranspose2d(curr_dim,
#                                    curr_dim // 2,
#                                    kernel_size=4,
#                                    stride=2,
#                                    padding=1,
#                                    bias=False),
#                 # nn.Upsample(scale_factor=(2, 2)),
#                 # nn.Conv2d(curr_dim, curr_dim//2, kernel_size=3, padding=1, bias=False),
#                 nn.InstanceNorm2d(curr_dim // 2, affine=False),
#                 nn.ReLU(inplace=True)
#             ]
#             curr_dim = curr_dim // 2

#         transform += [
#             nn.Conv2d(curr_dim,
#                       3,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1,
#                       bias=False)
#         ]
#         self.transform = nn.Sequential(*transform)

#     def forward(self, image, label_feature):
#         label_feature = self.fc(label_feature)
#         label_feature = label_feature.view(label_feature.size(0), self.nf, self.size, self.size)
#         label_feature = self.transform(label_feature)

#         # mixed_feature = label_feature + image
#         mixed_feature = torch.cat((label_feature, image), dim=1)
#         return mixed_feature

# class Discriminator(nn.Module):
#     """
#     Discriminator network with PatchGAN.
#     Reference: https://github.com/yunjey/stargan/blob/master/model.py
#     """
#     def __init__(self, num_classes, image_size=224, conv_dim=64, repeat_num=5):
#         super(Discriminator, self).__init__()
#         layers = []
#         layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
#         layers.append(nn.LeakyReLU(0.01))

#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
#             layers.append(nn.LeakyReLU(0.01))
#             curr_dim = curr_dim * 2

#         kernel_size = int(image_size / (2**repeat_num))
#         self.main = nn.Sequential(*layers)
#         self.fc = nn.Conv2d(curr_dim, num_classes + 1, kernel_size=kernel_size, bias=False)

#     def forward(self, x):
#         h = self.main(x)
#         out = self.fc(h)
#         return out.squeeze()








def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        #print(self.name)
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module):
        name = "weight"
        fn = SpectralNorm()

        try:
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")
        except AttributeError:
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            u = Parameter(w.data.new(height).normal_(0, 1),
                          requires_grad=False)
            v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            w_bar = Parameter(w.data)

            #del module._parameters[name]

            module.register_parameter(name + "_u", u)
            module.register_parameter(name + "_v", v)
            module.register_parameter(name + "_bar", w_bar)

        # remove w from parameter list
        del module._parameters[name]

        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_u']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_bar']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def spectral_norm(module):
    SpectralNorm.apply(module)
    return module


def remove_spectral_norm(module):
    name = 'weight'
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, label, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, real_label], dim=-1)
        else:
            fake_label = self.fake_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, fake_label], dim=-1)
        return target_tensor

    def __call__(self, prediction, label, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(label, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt["lr_policy"] == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count -
            #                  opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 1.0 - max(0, epoch - opt["epochs"]) / float(opt["decay"] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt["lr_policy"] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt["lr_policy"] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt["lr_policy"] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.n_epochs,
                                                   eta_min=0)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
