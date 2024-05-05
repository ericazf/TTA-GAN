import torch    
import torch.nn as nn 
import sys,os
import numpy as np
from torchvision import transforms
from torch.utils import data 
from torch.distributions import Categorical
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"]='1'
    
sys.path.append("../")
from utils.module import * 
from utils.tools import * 
from utils.dataloader import * 
from utils.backbone import * 

num_classes = {
    "cifar10-0": 10,
    "cifar10-1": 10,
    "cifar10-2": 10,
    "nus-wide": 21,
    "flickr25k":38,
    "imagenet100": 100
}

def get_config():
    config = { 
        "dataset": "flickr25k",
        "bits":32,
        "model_name":"VGG11",
        "save_path": "../save/DPSH",

        "code_save": "../code/DPSH",
        "anchor_save": "../anchorcode/DPSH",

        "epochs":1,
        "steps":300,
        "batch_size":12,
        "lr": 1e-4,
        "T":0.05,

        "iteration": 7,
        "epsilon": 8/255.0,
        "alpha": 2/255.0,
        "num": 2,
        "threshold":0.3
    }
    return config 

def circle_similarity(batch_feature, features, batch_label, labels, bit):
    similarity_matrix = batch_feature @ features.transpose(1, 0)
    similarity_matrix = similarity_matrix / bit
    label_matrix = (batch_label.mm(labels.t()) > 0)
    
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1) 
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S

def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(
        x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt

#--------------------------THA-----------------------------
class THA(nn.Module):
    def __init__(self, dbloader, num_db):
        super(THA, self).__init__()
        self.bits = config["bits"]
        self.T = config["T"]
        self.num_class = num_classes[config["dataset"]]
        
        self._build_model()
        # self.dB, self.db_labels = self.generate_code(dbloader, num_db)
        self.dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.dB = torch.from_numpy(self.dB)
        self.db_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.db_labels = torch.from_numpy(self.db_labels)

    def _build_model(self):
        self.prototype = PrototypeNet(self.bits, self.num_class).cuda()

        if config["model_name"].startswith("VGG"):
            self.hash_model = VGG(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("AlexNet"):
            self.hash_model = AlexNet(self.bits).cuda()
        elif config["model_name"].startswith("ResNet"):
            self.hash_model = ResNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("DenseNet"):
            self.hash_model = DenseNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("Incv3"):
            self.hash_model = InceptionV3(config["model_name"], self.bits).cuda()
        self.hash_model.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))))
        self.hash_model.eval()
        
    def hashloss(self, anchorcode, targetlabel, traincodes, trainlabels):
        S = (targetlabel.mm(trainlabels.t()) > 0).float()
        omiga = 1/2 * anchorcode.mm(traincodes.t())
        logloss = (torch.log(1 + torch.exp(-omiga.abs())) + omiga.clamp(min = 0) - S * omiga).sum()
        return logloss 

    def circleloss(self, anchorcode, targetlabel, traincodes, trainlabels):
        gamma = 1
        sp, sn = circle_similarity(anchorcode, traincodes, targetlabel, trainlabels, self.bits)
        ap = torch.clamp_min(- sp.detach() + 2, min=0.)
        an = torch.clamp_min(sn.detach() + 2, min=0.)

        logit_p = - ap * sp * gamma
        logit_n = an * sn * gamma
        loss = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
        return loss
    
    def train_prototype(self, train_loader, num_train, target_labels):
        optimizer_l = torch.optim.Adam(self.prototype.parameters(), lr=config["lr"], betas=(0.5, 0.999))
        # epochs = 100
        epochs = 1
        steps = 300
        batch_size = 64
        lr_steps = epochs * steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()

        B, train_labels = self.generate_code(train_loader, num_train)
        B = B.cuda()
        train_labels = train_labels.cuda()
 
        for epoch in range(epochs):
            for i in range(steps):
                select_index = np.random.choice(range(target_labels.size(0)), size=batch_size)
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()

                optimizer_l.zero_grad()
                S = CalcSim(batch_target_label, train_labels)
                _, target_hash_l, label_pred = self.prototype(batch_target_label)

                # logloss = self.circleloss(target_hash_l, batch_target_label, B, train_labels)/(batch_size)
                # regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size)
                # loss = logloss + regterm

                logloss = self.hashloss(target_hash_l, batch_target_label, B, train_labels)/(num_train * batch_size)
                regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size)
                classifer_loss = criterion_l2(label_pred, batch_target_label)
                loss = logloss + classifer_loss + regterm

                loss.backward()
                optimizer_l.step()
                if i % 100 == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, logloss:{:.5f}, regterm: {:.5f}'
                        .format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm))
                scheduler.step()
        # self.save_prototype()
        
    def test_prototype(self, total_labels):
        self.prototype.eval()
        label_size = total_labels.size(0)
        qB = torch.zeros(label_size, self.bits)

        for i in range(total_labels.size(0)):
            label = total_labels[i].cuda()
            _, code, _ = self.prototype(label)
            code = code.cpu().sign().data
            qB[i, :] = code

        map = CalcMap(qB, self.dB, total_labels, self.db_labels)
        print("prototypenet test map:{:.4f}".format(map))
        return map 

    def generate_code(self, trainloader, num_train):
        hashcode = torch.zeros(num_train, self.bits)
        hashlabels = torch.zeros(num_train, self.num_class)
        for iter, (img, label, idx) in enumerate(trainloader):
            img = img.cuda()
            label = label.cuda()
            H, code = self.hash_model(img)
            hashcode[idx, :] = code.cpu().data.sign()
            hashlabels[idx, :] = label.cpu().data
        return hashcode, hashlabels 
    
    def save_prototype(self):
        torch.save(self.prototype.state_dict(), os.path.join(config["anchor_save"], "proto_{}_{}_{}.pth".format(config["model_name"], config["dataset"], config["bits"])))

#----------------------CHCM---------------------
class CHCM(nn.Module):
    def __init__(self, dbloader, num_db):
        super(CHCM, self).__init__()
        self.bits = config["bits"]
        self.epochs = config["epochs"]
        self.steps = config["steps"]
        self.T = config["T"]
        self.num_class = num_classes[config["dataset"]]
        
        self._build_model()
        # self.dB, self.db_labels = self.generate_code(dbloader, num_db)
        self.dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.dB = torch.from_numpy(self.dB)
        self.db_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.db_labels = torch.from_numpy(self.db_labels)

    def _build_model(self):
        self.prototype = PrototypeNet(self.bits, self.num_class).cuda()

        if config["model_name"].startswith("VGG"):
            self.hash_model = VGG(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("AlexNet"):
            self.hash_model = AlexNet(self.bits).cuda()
        elif config["model_name"].startswith("ResNet"):
            self.hash_model = ResNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("DenseNet"):
            self.hash_model = DenseNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("Incv3"):
            self.hash_model = InceptionV3(config["model_name"], self.bits).cuda()
        self.hash_model.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))))
        self.hash_model.eval()

    def generate_code(self, trainloader, num_train):
        hashcode = torch.zeros(num_train, self.bits)
        hashlabels = torch.zeros(num_train, self.num_class)
        for iter, (img, label, idx) in enumerate(trainloader):
            img = img.cuda()
            label = label.cuda()
            H, code = self.hash_model(img)
            hashcode[idx, :] = code.cpu().data.sign()
            hashlabels[idx, :] = label.cpu().data
        return hashcode, hashlabels 
    
    def train(self, trainloader, num_train, target_labels):
        label_size = target_labels.size(0)
        aB = torch.zeros(label_size, self.bits)
        traincodes, trainlabels = self.generate_code(trainloader, num_train)
        for i in range(label_size):
            label = target_labels[i].unsqueeze(0)
            w = torch.sum(label * trainlabels, dim = 1)/torch.sum(torch.sign(label + trainlabels), dim = 1)
            w = w.unsqueeze(1)
            w1 = (w > 0).float()
            w2 = 1 - w1 
            c1 = traincodes.size(0)/w1.sum()
            c2 = traincodes.size(0)/w2.sum()   
            aB[i] = torch.sign(torch.sum(c1*w1*traincodes - c2*w2*traincodes, dim = 0))
        
        map = CalcMap(aB, self.dB, target_labels, self.db_labels)
        print("CHCM anchor t-MAP:{:.4f}".format(map))
        
        # anchor_path = "../anchorcode/CHCM"
        # if not os.path.exists(anchor_path):
        #     os.mkdir(anchor_path)
        # np.save(os.path.join(anchor_path, "AnchorCode_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])), aB.numpy())
        # np.save(os.path.join(anchor_path, "TargetLabel_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])), target_labels.numpy())
        return aB 
    
#-----------------------------------------IAO-------------------------------
class IAO(nn.Module):
    def __init__(self, dbloader, num_db):
        super(IAO, self).__init__()
        self.bits = config["bits"]
        self.epochs = config["epochs"]
        self.steps = config["steps"]
        self.T = config["T"]
        self.num_class = num_classes[config["dataset"]]
        
        self._build_model()
        # self.dB, self.db_labels = self.generate_code(dbloader, num_db)
        self.dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.dB = torch.from_numpy(self.dB)
        self.db_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.db_labels = torch.from_numpy(self.db_labels)

    def _build_model(self):
        self.prototype = PrototypeNet(self.bits, self.num_class).cuda()

        if config["model_name"].startswith("VGG"):
            self.hash_model = VGG(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("AlexNet"):
            self.hash_model = AlexNet(self.bits).cuda()
        elif config["model_name"].startswith("ResNet"):
            self.hash_model = ResNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("DenseNet"):
            self.hash_model = DenseNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("Incv3"):
            self.hash_model = InceptionV3(config["model_name"], self.bits).cuda()
        self.hash_model.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))))
        self.hash_model.eval()
        
    #modified version of circle loss
    def weightedtripletloss(self, anchorcode, targetlabel, traincodes, trainlabels):
        S = (targetlabel.mm(trainlabels.t()) > 0).float()
        omiga = anchorcode.mm(traincodes.t())/self.bits
        
        ap = torch.clamp_min(- omiga.detach() + 2, min=0.)
        an = torch.clamp_min(omiga.detach() + 2, min=0.)

        loss = torch.log(torch.exp(- ap * omiga * S).sum(dim = 1) - (1 - S).sum(dim = 1) + 1e-12) + torch.log(torch.exp(an * omiga * (1 - S)).sum(dim = 1) - S.sum(dim = 1) + 1e-12)
        return loss.sum()

    def train_code(self, train_loader, num_train, total_labels):
        B, L = self.generate_code(train_loader, num_train)
        B, L = B.cuda(), L.cuda()
        label_size = total_labels.size(0)
        qB = torch.zeros(label_size, self.bits)
      
        batch_size = 128
        if label_size % batch_size != 0:
            total = label_size // batch_size + 1
        else:
            total = label_size // batch_size 

        for idx in range(total):
            end_index = min((idx + 1)*batch_size, label_size)
            start_index = idx * batch_size
            target_label = total_labels[start_index:end_index, :].cuda()
            num = end_index - start_index

            logits = torch.randn(num, self.bits).cuda()
            logits.requires_grad = True 
            # eta = 1 for flickr25k and imagenet, 2 for nus-wide
            eta = 1
            optimizer = torch.optim.Adam([logits], lr = eta,  betas=(0.5, 0.999))
            steps = 1000
            
            alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
            for i in range(steps):
                optimizer.zero_grad()
                delta = alpha[int(i/steps * 10)]
                anchor_code = torch.tanh(logits * delta)

                logloss = self.weightedtripletloss(anchor_code, target_label, B, L)/batch_size
                loss = logloss 
                loss.backward()
                optimizer.step()
            
            map_idx = CalcMap(anchor_code.cpu().sign().data, self.dB, target_label.cpu().data, self.db_labels)
            print("idx:{}, map:{:.5f}".format(idx, map_idx))   
            
            anchor_code = anchor_code.cpu().sign().data
            qB[start_index:end_index, :] = anchor_code 
            
        map = CalcMap(qB, self.dB, total_labels, self.db_labels)
        # if not os.path.exists(config["anchor_save"]):
        #     os.mkdir(config["anchor_save"])
        # np.save(os.path.join(config["anchor_save"], "AnchorCode_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])), qB.numpy())
        # np.save(os.path.join(config["anchor_save"], "TargetLabel_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])), total_labels.numpy())
        print("map:{:.4f}".format(map))

    def test(self):
        qB = np.load(os.path.join(config["anchor_save"], "AnchorCode_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])))
        total_labels = np.load(os.path.join(config["anchor_save"], "TargetLabel_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])))
        qB = torch.from_numpy(qB)
        total_labels = torch.from_numpy(total_labels)

        map_ = CalcMap(qB, self.dB, total_labels, self.db_labels)
        print(map_)

    def generate_code(self, trainloader, num_train):
        hashcode = torch.zeros(num_train, self.bits)
        hashlabels = torch.zeros(num_train, self.num_class)
        for iter, (img, label, idx) in enumerate(trainloader):
            img = img.cuda()
            label = label.cuda()
            H, code = self.hash_model(img)
            hashcode[idx, :] = code.cpu().data.sign()
            hashlabels[idx, :] = label.cpu().data
        return hashcode, hashlabels 

    
if __name__ == "__main__":
    rand_idx = torch.randint(low = 0, high = 1000, size = (1,))[0]
    set_seed(rand_idx)
    config = get_config()

    data_path = "../data"
    img_dir = "/data1/zhufei/datasets"

    if config["model_name"] == "Incv3":
        scale_size = 300
        crop_size = 299
    else:
        scale_size = 256
        crop_size = 224

    if config["dataset"] == "nus-wide" or config["dataset"] == "flickr25k":
        transform = transforms.Compose(
        [
            transforms.Resize(scale_size),
            transforms.CenterCrop(crop_size),
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
                transforms.Resize(scale_size),
                transforms.CenterCrop(crop_size),
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

    iao = IAO(dbloader, num_db)
    iao.train_code(trainloader, num_train, unique_labels)

    # tha = THA(dbloader, num_db)
    # tha.train_prototype(trainloader, num_train, unique_labels)
    # tha.test_prototype(unique_labels)

    # total = 0
    # for i in range(len(unique_labels)):
    #     tha.train_prototype(trainloader, num_train, unique_labels[i,:].unsqueeze(0))
    #     map = tha.test_prototype(unique_labels[i,:].unsqueeze(0))
    #     total = total + map 
    # print(total/len(unique_labels))
    
    # chcm = CHCM(dbloader, num_db)
    # chcm.train(trainloader, num_train, unique_labels)