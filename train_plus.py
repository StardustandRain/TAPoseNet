from __future__ import print_function
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from model_plus import Model_Plus,TAligNet
from autoencoder import PointNetEncoder,AutoEncoder
import config.config as cfg
from dataset_plus import TrainingDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def add_self_loop(adjacency_matrix):
    """
    在邻接矩阵上加入自环

    参数：
    - adjacency_matrix: 原始的邻接矩阵

    返回：
    - adjacency_matrix_with_self_loop: 加入自环后的邻接矩阵
    """
    # 在对角线上加入自环
    adjacency_matrix_with_self_loop = adjacency_matrix + torch.eye(adjacency_matrix.size(0)).cuda()

    return adjacency_matrix_with_self_loop
def symmetric_normalization_with_self_loop(adjacency_matrix):
    """
    对称归一化邻接矩阵并加入自环

    参数：
    - adjacency_matrix: 原始的邻接矩阵

    返回：
    - normalized_adjacency: 对称归一化并加入自环后的邻接矩阵
    """
    # 加入自环
    adjacency_matrix_with_self_loop = add_self_loop(adjacency_matrix)

    # 计算度矩阵的逆平方根
    degree_matrix_inv_sqrt = torch.diag(torch.pow(torch.sum(adjacency_matrix_with_self_loop, dim=1), -0.5))

    # 对称归一化
    normalized_adjacency = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adjacency_matrix_with_self_loop), degree_matrix_inv_sqrt)

    return normalized_adjacency
def model_initial(model, model_name):
    # 加载预训练模型
    pretrained_dict = torch.load(model_name)["model"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dictf = {k.replace('module.', ""): v for k, v in pretrained_dict.items() if k.replace('module.', "") in model_dict}
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dictf)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print("over")
def train(args,global_matrix,local_matrix,colli_matrix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir =r"./dataset_whole"
    train_pose_dir="./dataset_whole_pose"
    train_obb_dir="./dataset_whole_obb"
    train_loader = DataLoader(TrainingDataset(train_dir,train_pose_dir,train_obb_dir), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Try to load models
    auto_encoder=AutoEncoder()
    encoder_path="./checkpoint/autoencoder_best_2100.pth"
    auto_encoder.load_state_dict(torch.load(encoder_path))
    encoder=auto_encoder.encoder
    encoder.cuda()
    model = TAligNet()
    model.load_state_dict(torch.load("./checkpoint_plus/teeth_alignment_1600.pth"))
    #tooth_assembler = Tooth_Assembler()
    #reconl1_loss = GeometricReconstructionLoss()

    # model_path="./checkpoint_landmark/teeth_alignment_3999.pth"
    # model_initial(model,model_path)

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD([{'params': model.local_fea.parameters(), 'lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        # opt = optim.SGD([
        #     {'params': model.teeth_fea.parameters(), 'lr': args.lr},
        #     {'params': model.global_fea.parameters(), 'lr': args.lr},
        #     {'params': model.output.parameters(), 'lr': args.lr}])

    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6, last_epoch = -1)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    model.cuda()
    model.train()
    #scaler = GradScaler()
    inter_nums = len(train_loader)
    pbar=tqdm(range(args.epochs))
    vis_cent_loss=[]
    vis_angle_loss=[]

    for i in pbar:
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        cent_loss = 0
        angle_loss = 0
        recon_loss=0
        obb_loss=0
        # for data, edges, label in train_loader:
        data_loader=iter(train_loader)
        for data_num in range(inter_nums):

            train_data,teeth_center, tweights, rweights,Gteeth_pose,Rteeth_pose,Gteeth_center,train_label= next(data_loader)
            train_data = train_data.cuda().float()
            #train_label = train_label.cuda().float()
            teeth_center = teeth_center.cuda().float()
            Gteeth_pose = Gteeth_pose.cuda().float()
            Rteeth_pose = Rteeth_pose.cuda().float()
            tweights = tweights.cuda().float()
            rweights = rweights.cuda().float()
            Gteeth_center=Gteeth_center.cuda().float()
            # Rteeth_obb=Rteeth_obb.cuda().float()
            # Gteeth_obb=Gteeth_obb.cuda().float()

            #
            weights = rweights -1 + tweights

            batch_size = train_data.size()[0]
            opt.zero_grad()
            geo_code=[]
            input_data=train_data-teeth_center#中心化

            for tid in range(input_data.shape[1]):
                embedding,_,_=encoder(input_data[:,tid,:,:].permute(0,2,1))
                geo_code.append(embedding)
            geo_code=torch.stack(geo_code,dim=1)
            pdofs, pcent= model(geo_code,teeth_center,Rteeth_pose,global_matrix,local_matrix,colli_matrix)
            # assembled = tooth_assembler(train_data, teeth_center, pdofs, pcent,Rteeth_pose, device)
            # dof_loss_ = torch.sum(torch.sum(F.smooth_l1_loss(pdofs, Gte, reduction= "none"), dim=-1) * rweights) / pdofs.shape[0]#没有问题，这两个矩阵本就应该互逆
            cent_loss_ = torch.sum(torch.sum(F.smooth_l1_loss(pcent, Gteeth_center, reduction= "none"), dim=-1) * tweights) / pcent.shape[0]
                    # gtrans_numpy = gtrans.detach().cpu().numpy()
            angle_loss_ = torch.sum((1-torch.sum(pdofs*Gteeth_pose, dim=-1))*rweights) / pdofs.shape[0]#余弦相似度
            #obb_loss_=torch.sum(torch.sum(F.smooth_l1_loss(final_obb.view(-1,28,24),Gteeth_obb.view(-1,28,24)),dim=-1))/Gteeth_obb.shape[0]
            # recon_loss_, c_loss_ = reconl1_loss(final_points, train_label, weights, device)
            # err_loss_=torch.sum(torch.norm(err,dim=-1,p=2))/err.shape[0]
            loss=cent_loss_+10*angle_loss_

                # loss=col_loss_

            loss.backward()
            opt.step()
            # loss.backward()
            # # Unscales gradients and calls
            # # or skips optimizer.step()
            # opt.step()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)

            count += batch_size
            train_loss += loss.item()
            cent_loss += cent_loss_.item()
            angle_loss += angle_loss_.item()
            #recon_loss+=recon_loss_.item()

        train_loss=train_loss/inter_nums
        cent_loss=cent_loss/inter_nums
        angle_loss=angle_loss/inter_nums
        #recon_loss=recon_loss/inter_nums
        vis_cent_loss.append(cent_loss)
        vis_angle_loss.append(angle_loss)
        # vis_recon_loss.append(recon_loss)

        outstr = ('epoch %d /%d, loss: %.3f, cent_loss: %.3f, angle_loss: %.6f' % (
                i, args.epochs,train_loss,cent_loss, angle_loss))
        pbar.set_description(outstr)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)

        if (i+1) % cfg.SAVE_MODEL == 0:
            torch.save(model.state_dict(), 'checkpoint_plus/teeth_alignment_' + str(i+1+1600)+ '.pth')
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5


    plt.plot(vis_cent_loss, label='Trans Loss', color='green')
    plt.plot(vis_angle_loss,label='Angle Loss',color='red')
    # plt.plot(vis_recon_loss, label='Recon Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Traning Loss Curve")
    plt.savefig("./loss_plus.png")


if __name__ == "__main__":
    #torch.backends.cudnn.enabled = False
    # Training settings
    parser = argparse.ArgumentParser(description='Teeth arrangement')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    # parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    global_matrix = torch.ones((28, 28)).cuda()
    local_matrix = torch.zeros((28, 28)).cuda()
    colli_matrix = torch.zeros((28, 28)).cuda()
    for i in range(28):
        local_matrix[i, i] = 1
        local_matrix[i][[(k - 1) for k in cfg.local_rela[i]]] = 1
        colli_matrix[i, i] = 1
        colli_matrix[i][[(k - 1) for k in cfg.colli_rela[i]]] = 1
    global_matrix = symmetric_normalization_with_self_loop(global_matrix)
    local_matrix = symmetric_normalization_with_self_loop(local_matrix)
    colli_matrix = symmetric_normalization_with_self_loop(colli_matrix)
    global_matrix = global_matrix.unsqueeze(0).repeat(4, 1, 1)
    local_matrix = local_matrix.unsqueeze(0).repeat(4, 1, 1)
    colli_matrix = colli_matrix.unsqueeze(0).repeat(4, 1, 1)
    train(args,global_matrix,local_matrix,colli_matrix)
