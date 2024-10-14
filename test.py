import os
import torch
import config.config as cfg
import numpy as np
import vtkmodules.all as vtk
from model_plus import TAligNet
from autoencoder import PointNetEncoder,AutoEncoder
from scipy.spatial.transform import Rotation
from train_plus import symmetric_normalization_with_self_loop

def read_stl(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader
def get_rotate_polydata(triangles, rpoints):
    points = vtk.vtkPoints()
    for p in rpoints:
        points.InsertNextPoint(p[0], p[1], p[2])

    new_plyd = vtk.vtkPolyData()
    new_plyd.SetPoints(points)
    new_plyd.SetPolys(triangles)

    return new_plyd
def write_stl(polydata, save_path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()

auto_encoder=AutoEncoder()
auto_encoder.eval()
encoder_path="./checkpoint/autoencoder_best_2100.pth"
auto_encoder.load_state_dict(torch.load(encoder_path))
encoder=auto_encoder.encoder
encoder.cuda()
model = TAligNet()
model.load_state_dict(torch.load("./checkpoint_align/teeth_alignment_3200.pth"))
model.cuda()
model.eval()
root_dir=r"C:\project\TANet\dataset_test"
root_pose_dir="./dataset_test_pose"
patient_dirs=os.listdir(root_dir)
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
global_matrix = global_matrix.unsqueeze(0)
local_matrix = local_matrix.unsqueeze(0)
colli_matrix = colli_matrix.unsqueeze(0)
os.makedirs("./result_plus",exist_ok=True)
for patient in patient_dirs:
    data_dir=os.path.join(root_dir,patient)
    pose_dir=os.path.join(root_pose_dir,patient)
    #obb_dir=os.path.join(root_obb_dir,patient)
    files=os.listdir(data_dir)
    teeth_points = np.zeros((len(files), cfg.sam_points, 3), np.float64)
    teeth_pose=np.zeros((len(files), 4), np.float64)
    #teeth_obb=np.zeros((len(files),8,3),np.float64)
    for file in files:
        data = os.path.join(data_dir, file)
        pose_data = os.path.join(pose_dir, file.replace(".stl", ".npy"))
        stl_reader = read_stl(data)
        teeth_nums = os.path.split(data)[-1].replace(".stl", "").split("_")[-1]  # 得到牙齿编号
        polydata = stl_reader.GetOutput()

        points = polydata.GetPoints()

        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        mesh_points = np.array(verts)
        # 随机采样1024个顶点
        se_index = np.random.randint(0, mesh_points.shape[0], 1024)
        se_points = mesh_points[se_index]
        index = int(cfg.INDEX_MINE[teeth_nums]) - 1
        teeth_points[index] = se_points
        pose = np.load(pose_data)
        teeth_pose[index] = pose[[1, 2, 3, 0]]
        teeth_pose[index][:3] = -teeth_pose[index][:3]  # 求逆变换
    teeth_points = teeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
    Rcp = np.mean(teeth_points, axis=0, keepdims=True)
    teeth_points = teeth_points - Rcp
    teeth_points = teeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)

    teeth_center = []
    for i in range(teeth_points.shape[0]):
        cenp = np.mean(teeth_points[i], axis=0)
        teeth_center.append(cenp)

    teeth_points = torch.unsqueeze(torch.tensor(np.array(teeth_points)), dim=0)
    teeth_center = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array(teeth_center)), dim=1), dim=0)
    teeth_pose=torch.unsqueeze(torch.tensor(teeth_pose),dim=0)
    test_data = teeth_points.cuda().float()
    teeth_center = teeth_center.cuda().float()
    teeth_pose=teeth_pose.cuda().float()

    geo_code=[]
    input_data=test_data-teeth_center
    with torch.no_grad():
        for tid in range(input_data.shape[1]):
            embedding, _, _ = encoder(input_data[:, tid, :, :].permute(0, 2, 1))
            geo_code.append(embedding)
        geo_code = torch.stack(geo_code, dim=1)
        pdofs, pcent= model(geo_code, teeth_center, teeth_pose,global_matrix,local_matrix,colli_matrix)
    pdofs = torch.squeeze(pdofs).detach().cpu().numpy()
    pcent = torch.squeeze(pcent).detach().cpu().numpy()
    teeth_center=torch.squeeze(torch.squeeze(teeth_center,dim=2)).detach().cpu().numpy()
    teeth_pose=torch.squeeze(teeth_pose).detach().cpu().numpy()
    rvappendFilter = vtk.vtkAppendPolyData()
    name=patient[8:11]
    for file in files:
        file=os.path.join(data_dir,file)
        teeth_nums = file.replace(".stl", "").split("_")[-1] # 得到牙齿编号
        index = int(cfg.INDEX_MINE[teeth_nums]) - 1
        stl_reader=read_stl(file)
        polydata = stl_reader.GetOutput()
        points = polydata.GetPoints()
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        rvpoints = np.array(verts)
        triangles = polydata.GetPolys()
        #mesh_points = mesh_points - Gacenp
        rcp = np.mean(rvpoints, axis=0)
        rvpoints = rvpoints - rcp
        original_rt = Rotation.from_quat(teeth_pose[index]).as_matrix()
        original_rt=np.linalg.inv(original_rt)
        rvpoints = np.dot(original_rt,rvpoints.T).T
        prt=Rotation.from_quat(pdofs[index]).as_matrix()
        rvpoints = np.dot(prt, rvpoints.T).T
        rvpoints = rvpoints + pcent[index]-teeth_center[index]#体现的是质心的位移
        rvpoints = rvpoints + rcp
        rvpolydata = get_rotate_polydata(triangles, rvpoints)
        rvappendFilter.AddInputData(rvpolydata)
        #write_stl(rvpolydata,f"./result_separate_rotate/{name}/teeth_{teeth_nums}.stl")
    rvappendFilter.Update()
    write_stl(rvappendFilter.GetOutput(), f"./result_plus/result_{patient}.stl")

