from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import vtkmodules.all as vtk
import config.config as cfg
import copy
from data.utils import rotate_maxtrix
from pytorch3d.transforms import *
from scipy.spatial.transform import Rotation
class TrainingDataset(Dataset):
    def __init__(self,root_dir,pose_dir,obb_dir):
        self.data_root=root_dir#牙齿数据目录
        self.dirs=os.listdir(root_dir)#病人目录
        self.pose_root=pose_dir#位姿数据目录
        self.obb_root=obb_dir

    def read_stl(self,file_path):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)
        reader.Update()

        return reader
    def __len__(self):
        return len(os.listdir(self.data_root))

    def __getitem__(self, idex):
        dir=os.path.join(self.data_root,self.dirs[idex])
        pose_dir=os.path.join(self.pose_root,self.dirs[idex])
        #obb_dir=os.path.join(self.obb_root,self.dirs[idex])
        files=os.listdir(dir)
        teeth_points = np.zeros((len(files), cfg.sam_points, 3), np.float64)
        teeth_pose=np.zeros((len(files),4),np.float64)#使用的是(x,y,z,w)
        #teeth_obb=np.zeros((len(files),8,3),np.float64)
        for file in files:
            data=os.path.join(dir,file)
            pose_data=os.path.join(pose_dir,file.replace(".stl",".npy"))
            stl_reader=self.read_stl(data)
            teeth_nums = os.path.split(data)[-1].replace(".stl", "").split("_")[-1]  # 得到牙齿编号
            polydata = stl_reader.GetOutput()

            points = polydata.GetPoints()#牙齿的全部点

            verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
            mesh_points = np.array(verts)
            # 随机采样1024个顶点
            se_index = np.random.randint(0, mesh_points.shape[0], 1024)
            se_points = mesh_points[se_index]

            pose=np.load(pose_data)

            index = int(cfg.INDEX_MINE[teeth_nums]) - 1
            teeth_points[index] = se_points
            teeth_pose[index]=pose[[1,2,3,0]]
            teeth_pose[index][:3]=-teeth_pose[index][:3]#求逆变换
            #得到obb框
            # rotation_matrix=Rotation.from_quat(pose[[1,2,3,0]]).as_matrix()
            # centroid=np.mean(mesh_points,axis=0)
            # mesh_points=mesh_points-centroid
            # rotated_points=np.dot(rotation_matrix,mesh_points.T).T
            # rotated_pcd=o3d.geometry.PointCloud()
            # rotated_pcd.points=o3d.utility.Vector3dVector(rotated_points)
            # aabb=rotated_pcd.get_axis_aligned_bounding_box()
            # obb_vertices=np.array(aabb.get_box_points())
            # rotation_matrix_v = np.linalg.inv(rotation_matrix)
            # obb_vertices = np.dot(rotation_matrix_v, obb_vertices.T).T + centroid
            #teeth_obb[index]=np.load(os.path.join(obb_dir,file.replace(".stl",".npy")))
        nums = teeth_points.shape[0]
        rotate_nums = np.random.randint(6, nums, 1)[0]
        rotate_index = [i for i in range(nums)]
        np.random.shuffle(rotate_index)
        rotate_index = rotate_index[0: rotate_nums]

        ###################ground tooth decentralization################
        Gteeth_points = copy.deepcopy(teeth_points).reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)  # 一整副牙齿的点云
        Gacenp = np.mean(Gteeth_points, axis=0, keepdims=True)  # 牙齿的中心
        Gteeth_points = Gteeth_points - Gacenp#以牙齿为中心
        Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
        #
        # Gteeth_obb=copy.deepcopy(teeth_obb).reshape(cfg.teeth_nums*8,cfg.dim)
        # Gteeth_obb=Gteeth_obb-Gacenp
        # Gteeth_obb=Gteeth_obb.reshape(cfg.teeth_nums,8,cfg.dim)

        Gteeth_pose=copy.deepcopy(teeth_pose)
        ###################ground tooth decentralization over############
        Rweights = np.ones((Gteeth_points.shape[0]))
        Tweights = np.ones((Gteeth_points.shape[0]))

        Rteeth_points = copy.deepcopy(Gteeth_points)
        Rteeth_pose=copy.deepcopy(Gteeth_pose)
        #Rteeth_obb=copy.deepcopy(Gteeth_obb)
        #rms = np.eye(3, 3).reshape(1, 3, 3).repeat(cfg.teeth_nums, axis=0)
        ###################tooth rotate################################
        for tid in rotate_index:
            v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
            cen = np.mean(Rteeth_points[tid], axis=0)
            points = Rteeth_points[tid] - cen
            #obb_vertices=Rteeth_obb[tid]-cen
            rotaxis = np.random.random(3) * 2 - 1 + 0.01
            rotaxis = rotaxis / np.linalg.norm(rotaxis)  # 单位向量
            angle_ = v1 * np.random.randint(0, 300, 1) / 10.0  # [-30°--30°]
            rt = rotate_maxtrix(rotaxis, angle_)
            rt = rt[0:3, 0:3]#旋转矩阵
            original_qua=Rteeth_pose[tid]
            # original_qua[:3]=-original_qua[:3]#求逆变换
            original_rt=Rotation.from_quat(original_qua).as_matrix()
            final_rt=np.dot(rt,original_rt)#这里的左乘以及右乘需要验证
            Rteeth_pose[tid]=Rotation.from_matrix(final_rt).as_quat()#(x,y,z,w)
            points_ = (rt.dot(points.T)).T
            Rteeth_points[tid] = points_ + cen

            # obb_vertices_=(rt.dot(obb_vertices.T)).T
            # Rteeth_obb[tid]=obb_vertices_+cen
            #rms[tid] = rt  # 旋转矩阵
            Rweights[tid] = Rweights[tid] + abs(angle_) * 3 / 100.0#变化越大，权重越大
        ###################tooth rotate over#############################

        ###################tooth translation#############################
        translate_nums = np.random.randint(6, nums, 1)[0]
        translate_index = [i for i in range(nums)]
        np.random.shuffle(translate_index)
        translate_index = translate_index[0: translate_nums]
        trans_v = np.array([[-2, -2, 2]])
        for i in range(Rteeth_points.shape[0]):
            index = np.random.randint(0, 3, 1)[0]
            # rotaxis = np.random.random(3) * 2 - 1 + 0.01
            rotaxis = cfg.ROTAXIS[index]  # rotaxis / np.linalg.norm(rotaxis)

            # v1 = np.random.normal(0, 1, size=(1))[0]
            # fg = np.clip(np.array([v1]), -1, 1)
            # trans_v = fg * rotaxis * scalev

            v1 = np.random.normal(0, 1, size=(1))[0]
            v2 = np.random.normal(0, 1, size=(1))[0]
            v3 = np.random.normal(0, 1, size=(1))[0]
            fg = np.clip(np.array([[v1, v2, v3]]), -1, 1)

            if i in translate_index:
                Rteeth_points[i] = Rteeth_points[i] + fg * trans_v
                #Rteeth_obb[i]=Rteeth_obb[i]+fg*trans_v

        ###################tooth translation over#############################

        Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
        Rcp = np.mean(Rteeth_points, axis=0, keepdims=True)
        Rteeth_points = Rteeth_points - Rcp
        Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
        # Rteeth_obb=Rteeth_obb.reshape(cfg.teeth_nums*8,cfg.dim)
        # Rteeth_obb=Rteeth_obb-Rcp
        # Rteeth_obb=Rteeth_obb.reshape(cfg.teeth_nums,8,cfg.dim)

        Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
        Gteeth_points = Gteeth_points - Rcp
        Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
        # Gteeth_obb=Gteeth_obb.reshape(cfg.teeth_nums*8,cfg.dim)
        # Gteeth_obb=Gteeth_obb-Rcp
        # Gteeth_obb=Gteeth_obb.reshape(cfg.teeth_nums,8,cfg.dim)

        ###################quaternion rotate matrix#############################
        # trans = Transform3d().compose(Rotate(torch.tensor(rms[:, 0:3, 0:3])))
        # final_trans_mat = trans.get_matrix()
        # dof = matrix_to_quaternion(final_trans_mat[:, 0:3, 0:3])
        ###################quaternion rotate matrix over#########################

        ###################translation matrix over################################
        #trans_mats = np.zeros((Rteeth_points.shape[0], 3), np.float64)
        for di in range(Rteeth_points.shape[0]):
            censd = np.mean(Gteeth_points[di], axis=0) - np.mean(Rteeth_points[di], axis=0)
            #trans_mats[di] = censd
            Tweights[di] = Tweights[di] + abs(np.sum(censd)) / 10.0
        ###################center point after transformation######################

        teeth_center = []
        for i in range(Rteeth_points.shape[0]):
            cenp = np.mean(Rteeth_points[i], axis=0)
            teeth_center.append(cenp)
        Gteeth_center=[]
        for i in range(Gteeth_points.shape[0]):
            cenp = np.mean(Gteeth_points[i], axis=0)
            Gteeth_center.append(cenp)

        Gteeth_points = torch.tensor(np.array(Gteeth_points))
        Rteeth_points = torch.tensor(np.array(Rteeth_points))
        teeth_center = torch.unsqueeze(torch.tensor(np.array(teeth_center)), dim=1)#(14,1,3)
        Gteeth_center = torch.tensor(np.array(Gteeth_center))  # (14,3)
        tweights=torch.tensor(Tweights)
        rweights=torch.tensor(Rweights)
        Gteeth_pose=torch.tensor(Gteeth_pose)
        Rteeth_pose=torch.tensor(Rteeth_pose)
        # Gteeth_obb=torch.tensor(Gteeth_obb)
        # Rteeth_obb=torch.tensor(Rteeth_obb)
        # gdofs=torch.tensor(dof)
        # trans_mats=torch.tensor(trans_mats)

        return Rteeth_points,teeth_center,tweights,rweights,Gteeth_pose,Rteeth_pose,Gteeth_center,Gteeth_points

if __name__=="__main__":
    dataset=TrainingDataset("./dataset_whole","./dataset_whole_pose","./dataset_whole_obb")
    loader=DataLoader(dataset,shuffle=True,batch_size=2,num_workers=0,drop_last=True)
    print(len(loader))
    data_loader=iter(loader)
    for i in range(3):
        train_data,teeth_center, tweights, rweights,Gteeth_pose,Rteeth_pose,Gteeth_center,Gteeth_points = next(data_loader)
        print(train_data.shape)
        print(teeth_center.shape)
        print(tweights.shape)
        print(rweights.shape)
        print(Gteeth_pose.shape)
        print(Rteeth_pose.shape)
        print(Gteeth_center.shape)
        print(Gteeth_points.shape)
