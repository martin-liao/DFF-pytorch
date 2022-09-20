from torch.utils.data import DataLoader,random_split
# import wandb
from loss.loss import reweighted_Cross_Entroy_Loss
from models.dff import DFF
from dataset.semantic_dataset import Dataset
import torch.optim as optim
from datetime import datetime
import os
import torch
from test import vis_pred

NUM_CLASS = 19

class Trainer():
    def __init__(self, 
    root_path : str = "../kitti-semantic/data_semantics",
    epoch : int = 1,
    val_percent : float = 0.1,
    batch_size : int = 2,
    shuffle = True,
    num_workers = 4,
    lr : float = 1e-3,
    momentum : float = 0.9,
    weight_decay: float = 1e-4,
    dataset : str = "cityscapes",
    mode : str = "train",
    device : str = "cuda",
    scale : list = [1.0],
    size : tuple = (256,384),
    loss_weight : dict = {'side5':1.0,'fuse':1.0},
    pretrained_path : str = "./checkpoint/dff_cityscapes_resnet101.pth.tar"):
        
        # load dataset
        self.dataset = Dataset(root_path,scale = scale,size = size,dataset = dataset,mode = mode)
        length = len(self.dataset.dataset)
        self.datasetname = dataset

        # random split dataset and prepare dataloader
        loader_args = {
            'batch_size' : batch_size,
            'shuffle' : shuffle,
            "num_workers" : num_workers
        }
        n_val = int(length * val_percent)
        n_train = length - n_val
        train_dataset,val_dataest = random_split(self.dataset,[n_train,n_val])
        self.trainloader = DataLoader(dataset = train_dataset,**loader_args)
        self.valloader = DataLoader(dataset = val_dataest,**loader_args)
        
        # load model
        self.model = DFF(nclass = NUM_CLASS,pretrained = True,device = device)

        # load pretrained model
        if pretrained_path is not None:
            pretrained_model = torch.load(pretrained_path)['state_dict']
            self.model.load_state_dict(pretrained_model)

        # load optimizer and loss function
        
        params_list = [ {'params': self.model.pretrained.parameters(), 'lr': lr},
                    {'params': self.model.EW1.parameters(), 'lr': lr * 10},
                    {'params': self.model.side1.parameters(), 'lr': lr * 10},
                    {'params': self.model.side2.parameters(), 'lr': lr * 10},
                    {'params': self.model.side3.parameters(), 'lr': lr * 10},
                    {'params': self.model.side5.parameters(), 'lr': lr * 10},
                    {'params': self.model.side5_ew.parameters(), 'lr': lr * 10}]
        

        self.optimizer = optim.SGD(
            params = params_list,
            lr = lr,
            momentum = momentum,
            weight_decay = weight_decay
            )

        self.loss_func = reweighted_Cross_Entroy_Loss()
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer = self.optimizer,mode = 'min')

        self.epoch = epoch
        self.device = device
        self.loss_weight = loss_weight
        self.mode = mode
    
        # load wandb utils
        # self.experiment = wandb.init(project = dataset + mode)
        # self.experiment.config.update(dict(
        #     epoch = epoch,batch_size = batch_size,learning_rate = lr,
        #     img_scale = scale,pretrained = pretrained_path))
        

    def train(self,epoch,cpt = "./checkpoint"):
        self.model.train()
        for j,(idxs,imgs,tars) in enumerate(self.trainloader):
            imgs = imgs.to(self.device)
            tars = tars.to(self.device)
            side5,fuse = self.model(imgs)
            loss = self.loss_weight['side5'] * self.loss_func(side5,tars) + \
                self.loss_weight['fuse'] * self.loss_func(fuse,tars)

            self.optimizer.zero_grad()
            loss.backward()
            now = datetime.now()
            log = "epoch = {:01d} | train: {} | Batch = [{:03d}/{:03d}] | loss = {:.4f} |"
            log = log.format(epoch,now.strftime("%c"),j,len(self.trainloader),loss.item())
            print(log)
            self.optimizer.step()

            # self.experiment.log({
            #     "loss":loss.item(),
            #     "step":j,
            #     "epoch":epoch
            # })


        if self.datasetname == "kitti":
            save_epoch = 1
        elif self.datasetname == "cityscapes":
            save_epoch = 1
        
        if epoch % save_epoch == 0:
            path = os.path.join(cpt,"%s_fine_%d.pth"%(self.datasetname,epoch))
            torch.save({'state_dict':self.model.state_dict(),
            'epoch' : self.epoch,
            'optimizer' : self.optimizer.state_dict()},
            f = path)
    
    def validation(self,epoch):
        self.model.eval()
        for i, (idxs,imgs,tars) in enumerate(self.valloader):
            imgs = imgs.to(self.device)
            tars = tars.to(self.device)
            with torch.no_grad():
                side5,fuse = self.model(imgs)
                fuse_np = fuse[0].sigmoid_().cpu().numpy()
                gt_np = tars[0].cpu().numpy()
                img_np = imgs[0].cpu().numpy()
                fuse_np = vis_pred(fuse_np,None)
                gt_np = vis_pred(gt_np,None)

                loss = self.loss_weight['side5'] * self.loss_func(side5,tars) + \
                    self.loss_weight['fuse'] * self.loss_func(fuse,tars)
                now = datetime.now()
                log = "epoch : {:01d} | valid: {} | Batch = [{:03d}/{:03d}] | loss = {:.4f} |"
                log = log.format(epoch,now.strftime("%c"),i,len(self.valloader),loss.item())
                print(log)

                # self.experiment.log({
                #         "loss":loss.item(),
                #         "image":img_np,
                #         "pred_edge":wandb.Image(fuse_np),
                #         "edge_gt":wandb.Image(gt_np),
                #         "lr":self.optimizer.param_groups[0]['lr'],
                #        "epoch":epoch
                #     })
                return loss
        

if __name__ == "__main__":
    # root_folder = "../cityscapes"
    root_folder = "../kitti-semantic/data_semantics"
    dataset = "kitti"
    mode = "train"
    trainer = Trainer(root_path = root_folder,dataset = dataset,mode = mode)
    for epoch in range(1,trainer.epoch + 1):
        trainer.train(epoch)
        val_loss = trainer.validation(epoch)
        trainer.lr_scheduler.step(val_loss)


    
