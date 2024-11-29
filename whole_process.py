import os
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
import config.config as cfg
import numpy as np
import vtkmodules.all as vtk
from model_plus import TAligNet
from autoencoder import PointNetEncoder,AutoEncoder
from scipy.spatial.transform import Rotation
from train_plus import symmetric_normalization_with_self_loop
import open3d as o3d

def read_obj(file_path):
    reader = vtk.vtkOBJReader()
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


def write_obj(polydata, save_path):
    """
    将给定的 VTK PolyData 保存为 .obj 文件。

    参数：
        polydata: VTK PolyData 对象
        save_path: 输出 .obj 文件路径
    """
    writer = vtk.vtkOBJWriter()  # 使用 vtkOBJWriter
    writer.SetFileName(save_path)
    writer.SetInputData(polydata)
    writer.Update()
    writer.Write()
def registration(source_vertices,target_vertices):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source_vertices)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target_vertices)
    # 配准
    trans_init = np.eye(4)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    # ICP 点对点配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, max_correspondence_distance=1.0, init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False), criteria=criteria
    )
    rotation_matrix = reg_p2p.transformation[:3, :3]
    return rotation_matrix

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
data_dir=r"C:\project\3D-Teeth-Reconstruction-from-Five-Intra-oral-Images-main\demo2\ref_mesh\1\byFDI"
reference_dir=r"C:\project\3D-Teeth-Reconstruction-from-Five-Intra-oral-Images-main\reference_pose"
files=os.listdir(data_dir)
teeth_points = np.zeros((len(files), cfg.sam_points, 3), np.float64)
teeth_pose=np.zeros((len(files), 4), np.float64)
    #teeth_obb=np.zeros((len(files),8,3),np.float64)
global_trans_matrix=np.array([[1,0,0],
                              [0,0,-1],
                              [0,1,0]])
for file in files:
    teeth_nums=file.replace(".obj","")[-2:]
    data = os.path.join(data_dir, file)
    mesh=trimesh.load_mesh(data)
    mesh_points=np.array(mesh.vertices)
    mesh_points=np.dot(global_trans_matrix,mesh_points.T).T#进行坐标系的转换

    center = np.mean(mesh_points, axis=0, keepdims=True)
    source_verts = mesh_points - center
    reference_points=np.load(os.path.join(reference_dir,f"{teeth_nums}.npy"))
    rotation_matrix=registration(source_verts,reference_points)
    pose = R.from_matrix(rotation_matrix.copy()).as_quat()
    se_index = np.random.randint(0, mesh_points.shape[0], 1024)
    se_points = mesh_points[se_index]
    index = int(cfg.INDEX_MINE[teeth_nums]) - 1
    teeth_points[index] = se_points
    teeth_pose[index] = pose
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
for file in files:
    file=os.path.join(data_dir,file)
    teeth_nums = file.replace(".obj","")[-2:]
    index = int(cfg.INDEX_MINE[teeth_nums]) - 1
    stl_reader=read_obj(file)
    polydata = stl_reader.GetOutput()
    points = polydata.GetPoints()
    verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    rvpoints = np.array(verts)
    rvpoints=np.dot(global_trans_matrix,rvpoints.T).T
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
    rvpoints=np.dot(global_trans_matrix.T,rvpoints.T).T
    rvpolydata = get_rotate_polydata(triangles, rvpoints)
    rvappendFilter.AddInputData(rvpolydata)
        # if patient=="zq":
        #     write_stl(rvpolydata,f"./dataset_test/zq_step/teeth_{teeth_nums}.stl")
    rvappendFilter.Update()
    write_obj(rvappendFilter.GetOutput(), "./recon.obj")

