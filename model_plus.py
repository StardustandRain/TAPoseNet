import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from pytorch3d.transforms import *
import math

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 完全平方展开

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)knn算法
        else:
            idx = knn(x[:, 3:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)

class SELayer(nn.Module):
    def __init__(self,channel=100,reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,n,c=x.size()
        y=self.avg_pool(x.permute(0,2,1)).view(b,c)
        y=self.fc(y).view(b,1,c)
        return x*y

class DGCNN_semseg_s3dis(nn.Module):
    def __init__(self, k=20,emb_dims=1024,drop_out=0.5):
        super(DGCNN_semseg_s3dis, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=drop_out)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x

class GraphConvolution(nn.Module):

    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=nn.Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias=nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self,input,adj):
        support=torch.matmul(input,self.weight)
        #print("support: ",support.shape)
        output=torch.matmul(adj,support)
        #print("median: ",output.shape)
        if self.bias is not None:
            return output+self.bias
        else:
            return output
class GCN(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout):
        super(GCN, self).__init__()

        self.gc1=GraphConvolution(nfeat,nhid)
        self.gc2=GraphConvolution(nhid,nclass)
        self.dropout=dropout
        self.relu=nn.ReLU()

    def forward(self,x,adj):
        x=F.relu(self.gc1(x,adj))
        x=F.dropout(x,self.dropout,training=self.training)
        x=self.gc2(x,adj)
        x=F.tanh(x)
        return x
class TAligNet(nn.Module):
    def __init__(self,embed_dim=100, depth=4, num_heads=4,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(TAligNet, self).__init__()
        self.fc1=nn.Linear(300,100)

        self.fc2=nn.Linear(100,100)
        self.fc3=nn.Linear(100,100)
        self.se_block=SELayer()
        self.global_gcn=GCN(107,128,100,0.5)
        self.local_gcn=GCN(107,128,100,0.5)
        self.colli_gcn=GCN(107,128,100,0.5)
        # self.teeth_blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(depth)])
        self.fc4=nn.Linear(2800,1000)
        self.fc5=nn.Linear(1000,196)
        # self.bn1=nn.BatchNorm1d(100)
        # self.bn2 = nn.BatchNorm1d(100)
        # self.bn3 = nn.BatchNorm1d(100)
        # self.bn4 = nn.BatchNorm1d(1000)
        #self.bn5 = nn.BatchNorm1d(196)
    def forward(self,x,teeth_center,teeth_pose,global_matrix,local_matrix,colli_matrix):
        x=torch.cat([x,teeth_center.squeeze(2),teeth_pose],dim=-1)
        global_features=self.global_gcn(x,global_matrix)
        local_features=self.local_gcn(x,local_matrix)
        colli_features=self.colli_gcn(x,colli_matrix)
        z=torch.cat([global_features,local_features,colli_features],dim=-1)

        z=F.relu(self.fc1(z))
        z_=z.clone()

        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z=self.se_block(z)+z_
        # z_=z.clone()
        # for blk in self.teeth_blocks:
        #     z = blk(z)
        # z=z+z_
        z=z.view(-1,2800)
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        z=z.view(-1,28,7)#(B,28,7)
        dof=z[:,:,3:]#(B,28,4)
        cent=z[:,:,:3]#(B,28,3)
        dof=F.normalize(dof,dim=-1)
        return dof,cent
class Model_Plus(nn.Module):
    def __init__(self):
        super(Model_Plus, self).__init__()
        self.coarse_model=TAligNet()
        self.linear1=nn.Linear(6144,256)
        self.linear2=nn.Linear(256,100)
        self.linear3=nn.Linear(100,3)
        self.label_emb=nn.Embedding(28,3)
        self.dgcnn=DGCNN_semseg_s3dis()
        self.teeth_blocks = nn.ModuleList([
            Block(dim=256, num_heads=4, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for i in range(4)])
    def forward(self,x,teeth_center,teeth_pose,teeth_points):
        dof,cent=self.coarse_model(x,teeth_center,teeth_pose)
        dof_=dof.clone().detach()
        cent_=cent.clone().detach()#切断梯度回传
        pred_matrices=torch.cat([quaternion_to_matrix(dof_[idx][:,[1,2,3,0]]).unsqueeze(0) for idx in range(dof_.shape[0])],dim=0)
        original_matrices = torch.cat([quaternion_to_matrix(teeth_pose[idx][:, [1, 2, 3, 0]]).unsqueeze(0) for idx in range(teeth_pose.shape[0])], dim=0)
        pred_points=torch.zeros(teeth_points.shape).cuda()
        label_index=torch.tensor([i for i in range(28)]).cuda()
        label_index=label_index.unsqueeze(0).repeat(teeth_points.shape[0],1)
        label_value=self.label_emb(label_index).unsqueeze(2).repeat(1,1,teeth_points.shape[2],1)#(B,28,1024,3)
        #得到变换后的牙齿点云
        for idx in range(x.shape[0]):#每个batch分开变换
            verts=teeth_points[idx,:,:,:]-teeth_center[idx,:,:,:]
            verts=torch.bmm(verts,original_matrices[idx])
            verts=torch.bmm(pred_matrices[idx],verts.permute(0,2,1)).permute(0,2,1)
            verts=verts+cent_[idx].unsqueeze(1)
            pred_points[idx]=verts

        final_points=pred_points.clone()#(B,28,1024,3)
        z=torch.cat([final_points,label_value],dim=-1)#(B,28,1024,6)
        z=z.view(-1,28,1024*6)
        z=self.linear1(z)
        z_=z.clone()
        for blk in self.teeth_blocks:
            z = blk(z)
        z=z+z_
        z=F.relu(self.linear2(z))
        err=self.linear3(z)

        # z=pred_obb.reshape(-1,28,24)
        # z=self.linear1(z)
        # z=z+self.linear2(z)
        # z=z.view(-1,2800)
        # z=self.linear3(z)
        # err=z.view(-1,28,3)
        cent=cent+err
        final_points=final_points+err.unsqueeze(2)
        return dof,cent,final_points,err
if __name__=="__main__":
    model=Model_Plus()
    model.cuda()
    x=torch.randn(2,28,100).cuda()
    teeth_pose=torch.randn(2,28,4).cuda()
    teeth_center=torch.randn(2,28,1,3).cuda()
    teeth_points=torch.randn(2,28,1024,3).cuda()
    dof,cent,final_points,err=model(x,teeth_center,teeth_pose,teeth_points)
    print(dof.shape)
    print(cent.shape)
    print(final_points.shape)
    print(err.shape)



