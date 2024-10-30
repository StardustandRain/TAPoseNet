import os
import torch
import numpy as np
import trimesh
from pose_estimation import Pose_Esitimation
import argparse
import vtkmodules.all as vtk
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# 定义拟合平面的方程
def plane_equation(params, points):
    a, b, c, d = params
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return a * x + b * y + c * z + d
# 定义拟合平面的误差
def plane_error(params, points):
    return np.sum((plane_equation(params, points) - 1)**2)

def read_stl(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader
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
def pose_transform(se_points,label):
    normal_vectors_x=[]
    normal_vectors_z=[]
    label_max=np.max(label[:,0])
    label_min=np.min(label[:,0])
    # print(label_max)
    # print(label_min)
    #x轴
    #确定正方向
    reference_vector=se_points[label[:,0]==label_max][0]-se_points[label[:,0]==label_min][0]
    for i in range(label_min,label_max):
        initial_params = [1, 1, 1, 1]
        points = se_points[label[:, 0] == i]
        # 使用最小化算法拟合平面
        result = minimize(plane_error, initial_params, args=(points,), method='L-BFGS-B')

        # 提取拟合得到的平面参数
        fitted_params = result.x

        # 提取法向量
        normal_vector = fitted_params[:3]
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        if np.dot(reference_vector,normal_vector)<0:
            normal_vector=-normal_vector
        normal_vectors_x.append(normal_vector)
    normal_vectors = np.array(normal_vectors_x)
    x_axis = np.mean(normal_vectors, axis=0)
    x_axis = x_axis / np.linalg.norm(x_axis)
    #print(x_axis)
    #z轴
    #确定正方向
    label_max=np.max(label[:,2])
    label_min=np.min(label[:,2])
    reference_vector=se_points[label[:,2]==label_max][0]-se_points[label[:,2]==label_min][0]
    for i in range(label_min,label_max):
        initial_params = [1, 1, 1, 1]
        points = se_points[label[:, 2] == i]
        # 使用最小化算法拟合平面
        result = minimize(plane_error, initial_params, args=(points,), method='L-BFGS-B')

        # 提取拟合得到的平面参数
        fitted_params = result.x

        # 提取法向量
        normal_vector = fitted_params[:3]
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        if np.dot(reference_vector,normal_vector)<0:
            normal_vector=-normal_vector
        normal_vectors_z.append(normal_vector)
    normal_vectors = np.array(normal_vectors_z)
    z_axis = np.mean(normal_vectors, axis=0)
    z_axis = z_axis / np.linalg.norm(z_axis)
    #print(z_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=0)
    return rotation_matrix
def estimate(args):
    teeth_dir=args.teeth_dir
    pose_dir=args.pose_dir
    template_dir="./template_pcd"
    os.makedirs(pose_dir,exist_ok=True)
    #load the model
    model_left=Pose_Esitimation(args)
    model_right=Pose_Esitimation(args)
    model_left.load_state_dict(torch.load("./checkpoint/teeth_pose_estimate_left.pth"))
    model_right.load_state_dict(torch.load("./checkpoint/teeth_pose_estimate_right.pth"))
    model_left.cuda()
    model_right.cuda()
    model_left.eval()
    model_right.eval()
    teeth_data=os.listdir(teeth_dir)
    for tooth in teeth_data:
        teeth_num=tooth.replace(".stl","")[-2:]
        path = os.path.join(teeth_dir, tooth)
        if int(teeth_num)%10<6:
            mesh=trimesh.load(path)
            source_verts=np.array(mesh.vertices)
            centroid = np.mean(source_verts, axis=0, keepdims=True)
            source_verts = source_verts - centroid
            target_verts=np.load(os.path.join(template_dir,f"{teeth_num}.npy"))
            rotation_matrix=registration(source_verts,target_verts)
            quaternion=R.from_matrix(rotation_matrix.copy()).as_quat()#(x,y,z,w)
            np.save(os.path.join(pose_dir,tooth.replace(".stl",".npy")),quaternion[[3,0,1,2]])
        else:
            trans_matrix = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ])
            mesh=trimesh.load(path)
            mesh_points=mesh.vertices
            centroid = np.mean(mesh_points, axis=0, keepdims=True)
            mesh_points = mesh_points - centroid
            se_index = np.random.randint(0, mesh_points.shape[0], 1024)
            stl_reader = read_stl(path)
            polydata = stl_reader.GetOutput()
            # 检查是否已有法向量，若没有则生成法向量
            if not polydata.GetPointData().GetNormals():
                normalGenerator = vtk.vtkPolyDataNormals()
                normalGenerator.SetInputData(polydata)
                normalGenerator.ComputePointNormalsOn()  # 计算点的法向量
                normalGenerator.SplittingOff()  # 关闭拆分，避免增加点的数量
                normalGenerator.Update()
                polydata = normalGenerator.GetOutput()
            # 提取点的法向量
            normals = polydata.GetPointData().GetNormals()
            normals = np.array([normals.GetTuple(i) for i in range(polydata.GetNumberOfPoints())])
            se_normals = normals[se_index]
            se_points = mesh_points[se_index]
            if int(teeth_num)//10==3 or int(teeth_num)//10==4:
                se_normals = np.dot(trans_matrix, se_normals.T).T
                se_points = np.dot(trans_matrix, se_points.T).T
            se_points = np.concatenate([se_normals, se_points], axis=1)
            teeth_points = torch.unsqueeze(torch.tensor(se_points), dim=0)
            teeth_points = teeth_points.cuda().float()
            if int(teeth_num)//10==1 or int(teeth_num)//10==4:
                model=model_left
            else:
                model=model_right
            with torch.no_grad():
                x_data, y_data, z_data = model(teeth_points)
            result_x = x_data.squeeze(0).permute(1, 0).detach().cpu().numpy()
            result_y = y_data.squeeze(0).permute(1, 0).detach().cpu().numpy()
            result_z = z_data.squeeze(0).permute(1, 0).detach().cpu().numpy()
            result_x = np.argmax(result_x, axis=1)
            result_y = np.argmax(result_y, axis=1)
            result_z = np.argmax(result_z, axis=1)
            label_data = np.stack([result_x, result_y, result_z], axis=1)
            predict_rm = pose_transform(se_points[:, 3:], label_data)
            if int(teeth_num) // 10 == 3 or int(teeth_num) // 10 == 4:
                predict_rm[0, :] = -predict_rm[0, :]
                predict_rm[2, :] = -predict_rm[2, :]
                predict_rm = np.dot(trans_matrix, predict_rm.T).T
            pose=R.from_matrix(predict_rm).as_quat()
            np.save(os.path.join(pose_dir,tooth.replace(".stl",".npy")),pose[[3,0,1,2]])
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pose estimation')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--teeth_dir', type=str, default="./dataset_test/zq",
                        help='The directory of teeth')
    parser.add_argument('--pose_dir', type=str, default="./dataset_test_pose/zq",
                        help='The target pose directory')
    args = parser.parse_args()
    estimate(args)



