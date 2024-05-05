import os 
import torch, sys     
import time
import numpy as np
from PIL import Image
import torch.nn as nn    

import random 
from torchvision import transforms
from torch.autograd import Variable   
import torchvision.transforms.functional as Fv
import scipy.stats as st

sys.path.append("../") 
from utils.module import * 
  
from utils.tools import *   
from utils.dataloader import * 
from utils.backbone import * 
os.environ["CUDA_VISIBLE_DEVICES"]='1,2'
   
def set_input_images(_input):
    _input = _input.cuda()
    _input = 2 * _input - 1
    return _input

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S


def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(
        x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt 

def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    padded = F.interpolate(padded, size = [img_size, img_size], mode = "bilinear", align_corners=False)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret 

# def gkern(kernlen=15, nsig=3):
#     x = np.linspace(-nsig, nsig, kernlen)
#     kern1d = st.norm.pdf(x)
#     kernel_raw = np.outer(kern1d, kern1d)
#     kernel = kernel_raw / kernel_raw.sum()
#     kernel = kernel.astype(np.float32)
#     gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
#     gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
#     gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
#     return gaussian_kernel
# T_kernel = gkern(7, 3)

def Rot(images, degree):
    if torch.rand(1) <= 1:
        angle = random.uniform(-degree, degree)
        result_images = Fv.rotate(images, angle)
    else:
        result_images = images 
    return result_images 

def get_config():
    config = {
        "dataset": "imagenet100",
        "model_name": "VGG11",
        "bits": 32,
        "save_path": "../save/DPSH",

        "target_path": "../save/DPSH",
        "target_model": "AlexNet_imagenet100_32_model.pth",
        
        "code_save": "../code/DPSH",
        "anchor_path": "../anchorcode/DPSH",
        "model_path": "../checkpoints/DPSH_Ens",
        "label_path": "../label/pn",
        
        "eps": 0.1,
        "epochs":50,
        "decay": 50,
        "batch_size": 24,
        "lr": 1e-4,
        "lr_policy": "linear",
        "threshold":0
    }
    return config 

#dr 100 89
class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size 

    def forward(self, images):
        pre_images = F.interpolate(images, size=[self.size, self.size], mode='bilinear', align_corners=False)
        return pre_images 

class TTAGAN_Ens(nn.Module):
    def __init__(self):
        super(TTAGAN_Ens, self).__init__()
        self.bits = config["bits"]
        classes_dic = {'flickr25k': 38, 'nus-wide':21, 'imagenet100': 100}
        self.num_classes = classes_dic[config["dataset"]]

        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.BCE = nn.BCELoss()
        self.eps = config["eps"]

        self._build_model()

    def _build_model(self):
        self.generator = nn.DataParallel(GeneratorCls(num_classes = self.num_classes)).cuda()
        self.discriminator = nn.DataParallel(MyDiscriminator(num_classes = self.num_classes)).cuda()
        
        self.VGG11 = VGG("VGG11", 32).cuda()
        self.VGG11.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format("VGG11", config["dataset"], 32))))
        self.ResNet50 = ResNet("ResNet50", 32).cuda()
        self.ResNet50.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format("ResNet50", config["dataset"], 32))))
        self.DenseNet161 = DenseNet("DenseNet161", 32).cuda()
        self.DenseNet161.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format("DenseNet161", config["dataset"], 32))))
        
        self.VGG11.eval()
        self.ResNet50.eval()
        self.DenseNet161.eval()
        self.criterionGAN = GANLoss('lsgan').cuda()

    def load_anchor(self):
        VGG11_code = np.load(os.path.join(config["anchor_path"], "AnchorCode_{}_{}_{}.npy".format("VGG11", config["dataset"], 32)))
        VGG11_labels = np.load(os.path.join(config["anchor_path"], "TargetLabel_{}_{}_{}.npy".format("VGG11", config["dataset"], 32)))
        ResNet50_code = np.load(os.path.join(config["anchor_path"], "AnchorCode_{}_{}_{}.npy".format("ResNet50", config["dataset"], 32)))
        ResNet50_labels = np.load(os.path.join(config["anchor_path"], "TargetLabel_{}_{}_{}.npy".format("ResNet50", config["dataset"], 32)))
        DenseNet161_code = np.load(os.path.join(config["anchor_path"], "AnchorCode_{}_{}_{}.npy".format("DenseNet161", config["dataset"], 32)))
        DenseNet161_labels = np.load(os.path.join(config["anchor_path"], "TargetLabel_{}_{}_{}.npy".format("DenseNet161", config["dataset"], 32)))
        
        self.label2idx = {}
        for num, label in enumerate(VGG11_labels):
            label_str = self.label2str(label)
            self.label2idx[label_str] = num 

        self.VGG11_code = torch.from_numpy(VGG11_code)
        self.ResNet50_code = torch.from_numpy(ResNet50_code)
        self.DenseNet161_code = torch.from_numpy(DenseNet161_code)

    def label2str(self, label):
        label_str = ""
        for i in label:
            label_str = label_str + str(int(i))
        return label_str

    def get_anchor(self, target_labels):
        batch_size = target_labels.size(0)
        Vcodes = torch.zeros(batch_size, self.bits)
        Rcodes = torch.zeros(batch_size, self.bits)
        Dcodes = torch.zeros(batch_size, self.bits)
        for i in range(batch_size):
            label = target_labels[i]
            label_str = self.label2str(label)
            Vcodes[i] = self.VGG11_code[self.label2idx[label_str]]
            Rcodes[i] = self.ResNet50_code[self.label2idx[label_str]]
            Dcodes[i] = self.DenseNet161_code[self.label2idx[label_str]]
        return Vcodes.cuda(), Rcodes.cuda(), Dcodes.cuda()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_hash_code(self, data_loader, num_data, model):
        B = torch.zeros(num_data, self.bits)
        labels = torch.zeros(num_data, self.num_classes)
        for it, (img, label, idx) in enumerate(data_loader, 0):
            with torch.no_grad():
                img = img.cuda()
                _, output = model(img)
                B[idx, :] = output.cpu().data.sign()
                labels[idx, :] = label 
        return B, labels

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if config["lr_policy"] == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()

        self.lr = self.optimizers[0].param_groups[0]['lr']

    def gradient_penalty(self, y, x):
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

#--------------------------------------------------------------
    def agg_loss(self, adv_img, target_hash_l, model):
        grad_total = 0 
        N = 3
        for i in range(N):
            _, target_hashing_g = model(Rot(DI(adv_img), 10))
            aug_loss = self.oriloss(target_hashing_g, target_hash_l)
            grad = torch.autograd.grad(aug_loss, adv_img, create_graph = False, retain_graph = False)[0]
            grad_total += grad 

        grad_total = grad_total / N
        return (grad_total.detach() * adv_img).sum()

    def oriloss(self, hash_code, target_hash):
        logloss = hash_code * target_hash
        logloss = torch.mean(logloss)
        logloss = (-logloss + 1)
        return logloss 
    
    def train(self, target_labels, train_loader, train_labels, database_loader, database_labels, test_loader, test_labels, num_database, num_train, num_test):
        # hyper = [0.5, 1, 10, 10, 10, 20, 40, 60, 80, 100]
        self.load_anchor()
        mse_loss = torch.nn.MSELoss()
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=config["lr"], betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=config["lr"], betas=(0.5, 0.999))
        self.optimizers = [optimizer_g, optimizer_d]
        self.schedulers = [get_scheduler(opt, config) for opt in self.optimizers]

        total_epochs = config["epochs"] + config["decay"]
        for epoch in range(total_epochs):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print("{:d}/{:d}, time:{}, learning rate:{:.7f}".format(epoch, total_epochs, current_time, self.lr))
            
            for i, data in enumerate(train_loader):
                real_input, batch_label, batch_ind = data
                real_input = real_input.cuda()
                batch_label = batch_label.cuda()
  
                select_index = np.random.choice(range(target_labels.size(0)), size=batch_label.size(0))
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()
                
                Vcodes, Rcodes, Dcodes = self.get_anchor(batch_target_label)
                adv_img, _ = self.generator(real_input, batch_target_label)
                adv_img = (adv_img + 1)/2

                # update D 
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    real_src, _ = self.discriminator(real_input)
                    fake_src, _ = self.discriminator(adv_img.detach())
                    real_d_loss = mse_loss(real_src, torch.zeros(real_src.size()).cuda())/38 
                    fake_d_loss = mse_loss(fake_src, torch.ones(fake_src.size()).cuda())/38

                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()

                # update G
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()

                fake_g_src, _ = self.discriminator(adv_img)
                fake_g_loss = mse_loss(fake_g_src, torch.zeros(fake_g_src.size()).cuda())/38
                reconstruction_loss = mse_loss(adv_img, real_input)


                with torch.no_grad():
                    _, target_hashing_g = self.VGG11(adv_img)
                    testloss = self.oriloss(target_hashing_g, Vcodes)

                logloss = self.agg_loss(adv_img, Vcodes, self.VGG11) + self.agg_loss(adv_img, Rcodes, self.ResNet50) + self.agg_loss(adv_img, Dcodes, self.DenseNet161)

                # backpropagation
                g_loss = (1/3)*logloss + fake_g_loss + 200 * reconstruction_loss
                #imagenet 
                # g_loss = (1/3)*logloss + fake_g_loss + hyper[int(epoch/10)] * reconstruction_loss 
                g_loss.backward()
                optimizer_g.step()
   
                if i % 30 == 0:
                    print('step:{} g_loss:{:.3f} d_loss:{:.3f} hash_loss:{:.3f} r_loss:{:.5f}'
                        .format(i, fake_g_loss, d_loss, testloss, reconstruction_loss))
        
            self.update_learning_rate()
            
        self.save_generator()
        self.save_discriminator()

    def test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
        self.generator.eval()

        qB = torch.zeros(num_test, self.bits)
        aB = torch.zeros(num_test, self.bits)
        targeted_labels = torch.zeros(num_test, self.num_classes)

        perceptibility = 0
        start = time.time()
        for it, data in enumerate(test_loader):
            data_input, _, data_ind = data

            select_index = np.random.choice(range(target_labels.size(0)), size=data_ind.size(0))
            batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
            targeted_labels[data_ind.numpy(), :] = batch_target_label

            with torch.no_grad():
                # data_input = set_input_images(data_input)
                data_input = data_input.cuda()

                anchor_code, Rcodes, Dcodes = self.get_anchor(batch_target_label.cuda())
                target_fake, _ = self.generator(data_input, batch_target_label.cuda())
                target_fake = (target_fake + 1)/2

                _, target_hashing = self.VGG11(target_fake)
                qB[data_ind.numpy(), :] = torch.sign(target_hashing.cpu().data)
                aB[data_ind.numpy(), :] = anchor_code.cpu().data

            perceptibility += F.mse_loss(data_input, target_fake).data * data_ind.size(0)

        end = time.time()
        print('Running time: %s Seconds'%(end-start))
        dB,_ = self.generate_hash_code(database_loader, num_database, self.VGG11)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        t_map = CalcMap(qB, dB, targeted_labels, database_labels)
        print('t_MAP(retrieval database): %3.5f' % (t_map))
        anchor_map = CalcMap(aB, dB, targeted_labels, database_labels)
        print('Anchor t_MAP:{:.5f}'.format(anchor_map))

#--------------------------------------------------------------

    def generate_code(self, data_loader, num_data):
        B = torch.zeros(num_data, self.bits)
        labels = torch.zeros(num_data, self.num_classes)
        for it, (img, label, idx) in enumerate(data_loader, 0):
            with torch.no_grad():
                img = img.cuda()
                _, output = self.hashing_model(img)
                B[idx, :] = output.cpu().data.sign()
                labels[idx, :] = label 
        return B, labels

    def save_generator(self):
        if not os.path.exists(config["model_path"]):
            os.makedirs(config["model_path"])
        torch.save(self.generator.module.state_dict(),
            os.path.join(config["model_path"], 'ensgenerator_{}_{}_{}.pth'.format(config["model_name"], config["dataset"], config["bits"])))

    def save_discriminator(self):
        if not os.path.exists(config["model_path"]):
            os.makedirs(config["model_path"])
        torch.save(self.discriminator.module.state_dict(),
            os.path.join(config["model_path"], "ensdiscriminator_{}_{}_{}.pth".format(config["model_name"], config["dataset"], config["bits"])))

    def load_generator(self):
        self.generator.module.load_state_dict(
            torch.load(os.path.join(config["model_path"], 'ensgenerator_{}_{}_{}.pth'.format(config["model_name"], config["dataset"], config["bits"]))))

    def load_discriminator(self):
        self.discriminator.load_state_dict(
            torch.load(os.path.join(config["model_path"], "ensdiscriminator_{}_{}_{}.pth".format(config["model_name"], config["dataset"], config["bits"])))
        )

    def load_model(self):
        self.load_generator()
        # self.load_discriminator()

    def cross_network_test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
        self.transfer_hashing_model.eval()
        self.generator.eval()
        qB = torch.zeros(num_test, self.transfer_bit)
        oB = torch.zeros(num_test, self.transfer_bit)
        targeted_labels = torch.zeros(num_test, self.num_classes)

        #----------------------------load database code---------------------------------
        method = config["target_model"].split("_")[0]
        dataset = config["target_model"].split("_")[1]
        bit = config["target_model"].split("_")[2]
        if not os.path.exists(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(method, dataset, bit))):
            dB, database_labels = self.generate_hash_code(database_loader, num_database, self.transfer_hashing_model, self.transfer_bit)
            np.save(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(method, dataset, bit)), dB.numpy())
            np.save(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(method, dataset, bit)), database_labels.numpy())
        else:
            print("--------")
            dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(method, dataset, bit)))
            database_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(method, dataset, bit)))
            dB = torch.from_numpy(dB)
            database_labels = torch.from_numpy(database_labels)

        #----------------------------load target label------------------------------
        if os.path.exists(os.path.join(config["label_path"], "{}.npy".format(config["dataset"]))):
            print("--------")
            targeted_labels = np.load(os.path.join(config["label_path"], "{}.npy".format(config["dataset"])))
            targeted_labels = torch.from_numpy(targeted_labels)

        perceptibility = 0
        start = time.time()
        for it, data in enumerate(test_loader):
            data_input, data_label, data_ind = data

            select_index = np.random.choice(range(target_labels.size(0)), size=data_ind.size(0))
            batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
            targeted_labels[data_ind, :] = batch_target_label

            if os.path.exists(os.path.join(config["label_path"], "{}.npy".format(config["dataset"]))):
                batch_target_label = targeted_labels[data_ind]
            else:
                select_index = np.random.choice(range(target_labels.size(0)), size=data_ind.size(0))
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
                targeted_labels[data_ind, :] = batch_target_label

            with torch.no_grad():
                # data_input = set_input_images(data_input)
                data_input = data_input.cuda()
                target_fake, _ = self.generator(data_input, batch_target_label.cuda())
                target_fake = (target_fake + 1)/2

                # perturbations = target_fake - data_input 
                # per = perturbations.view(perturbations.size(0), perturbations.size(1)*perturbations.size(2)*perturbations.size(3))
                # weight = (0.032 * math.sqrt(3*224*224))/torch.norm(per, dim = 1).view(per.size(0), 1, 1, 1)
                # target_fake = data_input + weight * perturbations
                perceptibility += F.mse_loss(data_input, target_fake).data * data_ind.size(0)

                _, target_hashing = self.transfer_hashing_model(target_fake)
                _, ori_hashing = self.transfer_hashing_model(data_input)
                qB[data_ind, :] = torch.sign(target_hashing.cpu().data)
                oB[data_ind, :] = torch.sign(ori_hashing.cpu().data)

        end = time.time()
        print('Running time: %s Seconds'%(end-start))
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        if config["dataset"] == "imagenet100":
            t_map = CalcTopMap(qB, dB, targeted_labels, database_labels, 1000)
            ori_tmap = CalcTopMap(oB, dB, targeted_labels, database_labels, 1000)
        else:
            t_map = CalcMap(qB, dB, targeted_labels, database_labels)
            ori_tmap = CalcMap(oB, dB, targeted_labels, database_labels)
        print('Adv t_MAP(retrieval database): %3.4f' % (t_map))
        print('Ori t_MAP(retrieval database): %3.4f' % (ori_tmap))

    def transfer_test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test, target_model_path):
        model_name = target_model_path.split("_")[0]
        self.transfer_bit = int(target_model_path.split("_")[-2])

        if model_name.startswith("VGG"):
            self.transfer_hashing_model = VGG(model_name, self.transfer_bit).cuda()
        elif model_name.startswith("AlexNet"):
            self.transfer_hashing_model = AlexNet(self.transfer_bit).cuda()
        elif model_name.startswith("ResNet"):
            self.transfer_hashing_model = ResNet(model_name, self.transfer_bit).cuda()
        elif model_name.startswith("Incv3"):
            self.transfer_hashing_model = InceptionV3(model_name, self.transfer_bit).cuda()
        elif model_name.startswith("DenseNet"):
            self.transfer_hashing_model = DenseNet(model_name, self.transfer_bit).cuda()

        self.transfer_hashing_model.load_state_dict(torch.load(os.path.join(config["target_path"], target_model_path)))
        self.transfer_hashing_model.eval()

        if model_name.startswith("Incv3"):
            self.transfer_hashing_model = nn.Sequential(
                Resize(299),
                self.transfer_hashing_model
            )
        self.cross_network_test(target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test)

if __name__ == "__main__":
    set_seed(100)
    config = get_config()

    data_path = "../data"
    img_dir = "/data1/zhufei/datasets"

    if config["dataset"] == "nus-wide" or config["dataset"] == "flickr25k":
        transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        train_dataset = MultiDatasetLabel(os.path.join(data_path, config["dataset"], "train_img.txt"),
                                    os.path.join(data_path, config["dataset"], "train_label.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        test_dataset = MultiDatasetLabel(os.path.join(data_path, config["dataset"], "test_img.txt"),
                                    os.path.join(data_path, config["dataset"], "test_label.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        database_dataset = MultiDatasetLabel(os.path.join(data_path, config["dataset"], "database_img.txt"),
                                os.path.join(data_path, config["dataset"], "database_label.txt"),
                                os.path.join(img_dir, config["dataset"]),
                                transform)
        train_label = train_dataset.get_label()
        test_label = test_dataset.get_label()
        db_label = database_dataset.get_label()
        unique_labels = db_label.unique(dim = 0)
    
    elif config["dataset"] == "imagenet100":
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        train_dataset = Imagenet(os.path.join(data_path, config["dataset"], "train.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        test_dataset = Imagenet(os.path.join(data_path, config["dataset"], "test10.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        database_dataset = Imagenet(os.path.join(data_path, config["dataset"], "database.txt"),
                                os.path.join(img_dir, config["dataset"]),
                                transform)
        train_label = train_dataset.get_label()
        test_label = test_dataset.get_label()
        db_label = database_dataset.get_label()
        unique_labels = db_label.unique(dim = 0)

    num_train, num_test, num_db = len(train_dataset), len(test_dataset), len(database_dataset)
    trainloader = data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers = 4)
    testloader = data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
    dbloader = data.DataLoader(database_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
 
    unique_labels = db_label.unique(dim = 0)
    pros_gan = TTAGAN_Ens()
    pros_gan.train(unique_labels, trainloader, train_label, dbloader, db_label, testloader, test_label, num_db, num_train, num_test)
    pros_gan.test(unique_labels, dbloader, testloader, db_label, test_label, num_db, num_test)
    
    # pros_gan.load_model()
    # pros_gan.transfer_test(unique_labels, dbloader, testloader, db_label, test_label, num_db, num_test, config["target_model"])
    