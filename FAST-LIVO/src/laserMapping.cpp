// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
// #include <common_lib.h>   // SparseMap 所在的头文件
#include <image_transport/image_transport.h>
#include "IMU_Processing.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <fast_livo/States.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vikit/camera_loader.h>
#include "lidar_selection.h"

#ifdef USE_ikdtree
#ifdef USE_ikdforest
#include <ikd-Forest/ikd_Forest.h>
#else

#include <ikd-Tree/ikd_Tree.h>

#endif
#else
#include <pcl/kdtree/kdtree_flann.h>
#endif

#define INIT_TIME (0.5)
#define MAXN (360000)
#define PUBFRAME_PERIOD (20)

float DET_RANGE = 300.0f;
#ifdef USE_ikdforest
const int laserCloudWidth = 200;
const int laserCloudHeight = 200;
const int laserCloudDepth = 200;
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;
#else
const float MOV_THRESHOLD = 1.5f;
#endif

mutex mtx_buffer;
condition_variable sig_buffer;

// mutex mtx_buffer_pointcloud;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, img_topic, config_file;

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);
// Vector3d Lidar_offset_to_IMU(0.04165, 0.02326, -0.0284); // Avia
Vector3d Lidar_offset_to_IMU;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0,
    effct_feat_num = 0, time_log_counter = 0, publish_count = 0;
int MIN_IMG_COUNT = 0;

double res_mean_last = 0.05;
//double gyr_cov_scale, acc_cov_scale;
double gyr_cov_scale = 0, acc_cov_scale = 0;
//double last_timestamp_lidar, last_timestamp_imu = -1.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
//double filter_size_corner_min, filter_size_surf_min, filter_size_map_min, fov_deg;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
//double cube_len, HALF_FOV_COS, FOV_DEG, total_distance, lidar_end_time, first_lidar_time = 0.0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
double first_img_time = -1.0;
//double kdtree_incremental_time, kdtree_search_time;
double kdtree_incremental_time = 0, kdtree_search_time = 0, kdtree_delete_time = 0.0;
int kdtree_search_counter = 0, kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;

//double copy_time, readd_time, fov_check_time, readd_box_time, delete_box_time;
double copy_time = 0, readd_time = 0, fov_check_time = 0, readd_box_time = 0, delete_box_time = 0;
double T1[MAXN], T2[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN];

double match_time = 0, solve_time = 0, solve_const_H_time = 0;

bool lidar_pushed, flg_reset, flg_exit = false;
bool ncc_en;
int dense_map_en = 1;
int img_en = 1;
int lidar_en = 1;
int debug = 0;
bool fast_lio_is_ready = false;
int grid_size, patch_size;
double outlier_threshold, ncc_thre;

vector<BoxPointType> cub_needrm;
vector<BoxPointType> cub_needad;
// deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;   // 收到的lidar点云
deque<double> time_buffer;   // Lidar消息时间戳
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<cv::Mat> img_buffer;   // 收到的图像信息
deque<double> img_time_buffer;  // 图像时间戳
vector<bool> point_selected_surf;
vector<vector<int>> pointSearchInd_surf;
vector<PointVector> Nearest_Points;
vector<double> res_last;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> cameraextrinT(3, 0.0);
vector<double> cameraextrinR(9, 0.0);
double total_residual;
double LASER_POINT_COV, IMG_POINT_COV, cam_fx, cam_fy, cam_cx, cam_cy;
bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
//surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr cube_points_add(new PointCloudXYZI());
PointCloudXYZI::Ptr map_cur_frame_point(new PointCloudXYZI());
PointCloudXYZI::Ptr sub_map_cur_frame_point(new PointCloudXYZI());  //; 当前图像用的视觉地图点，很稀疏的点云

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

#ifdef USE_ikdtree
#ifdef USE_ikdforest
KD_FOREST ikdforest;
#else
KD_TREE ikdtree;
#endif
#else
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
#endif

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
Eigen::Matrix3d Rcl;
Eigen::Vector3d Pcl;

//estimator inputs and output;
LidarMeasureGroup LidarMeasures;   //; 同步之后的消息
// SparseMap sparse_map;
#ifdef USE_IKFOM
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;
#else
StatesGroup state;
#endif

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

//; 这个应该是lidar的前端特征点云的预处理类，主要是对lidar的点云进行稍微的处理（但是和提取平面特征又不完全一样）
shared_ptr<Preprocess> p_pre(new Preprocess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)
{
#ifdef USE_IKFOM
    //state_ikfom write_state = kf.get_x();
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", LidarMeasures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
#else
    V3D rot_ang(Log(state.rot_end));
    fprintf(fp, "%lf ", LidarMeasures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state.pos_end(0), state.pos_end(1), state.pos_end(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
    fprintf(fp, "%lf %lf %lf ", state.vel_end(0), state.vel_end(1), state.vel_end(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
    fprintf(fp, "%lf %lf %lf ", state.bias_g(0), state.bias_g(1), state.bias_g(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state.bias_a(0), state.bias_a(1), state.bias_a(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state.gravity(0), state.gravity(1), state.gravity(2)); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
#endif
}

#ifdef USE_IKFOM
//project the lidar scan to world frame
void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}
#endif

// 把局部坐标系下的点，通过全局维护的位姿变量投影到世界坐标系下
void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
#ifdef USE_IKFOM
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
#else
    V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
#endif

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
#ifdef USE_IKFOM
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
#else
    V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
#endif
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
#ifdef USE_IKFOM
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
#else
    V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
#endif
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - floor(intensity);

    int reflection_map = intensity * 10000;
}

#ifndef USE_ikdforest
int points_cache_size = 0;

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    points_cache_size = points_history.size();
}

#endif

BoxPointType get_cube_point(float center_x, float center_y, float center_z)
{
    BoxPointType cube_points;
    V3F center_p(center_x, center_y, center_z);
    // cout<<"center_p: "<<center_p.transpose()<<endl;

    for (int i = 0; i < 3; i++)
    {
        cube_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
        cube_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
    }

    return cube_points;
}

BoxPointType get_cube_point(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax)
{
    BoxPointType cube_points;
    cube_points.vertex_max[0] = xmax;
    cube_points.vertex_max[1] = ymax;
    cube_points.vertex_max[2] = zmax;
    cube_points.vertex_min[0] = xmin;
    cube_points.vertex_min[1] = ymin;
    cube_points.vertex_min[2] = zmin;
    return cube_points;
}

#ifndef USE_ikdforest
BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;

//; 过滤在当前LiDAR的FOV内的点云
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
#ifdef USE_IKFOM
    //state_ikfom fov_state = kf.get_x();
    //V3D pos_LiD = fov_state.pos + fov_state.rot * fov_state.offset_T_L_I;
    V3D pos_LiD = pos_lid;
#else
    V3D pos_LiD = state.pos_end;
#endif
    if (!Localmap_Initialized)
    {
        //if (cube_len <= 2.0 * MOV_THRESHOLD * DET_RANGE) throw std::invalid_argument("[Error]: Local Map Size is too small! Please change parameter \"cube_side_length\" to larger than %d in the launch file.\n");
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // printf("Local Map is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", LocalMap_Points.vertex_min[0],LocalMap_Points.vertex_max[0],LocalMap_Points.vertex_min[1],LocalMap_Points.vertex_max[1],LocalMap_Points.vertex_min[2],LocalMap_Points.vertex_max[2]);
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
            dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                         double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
            // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
            // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
    //    printf("Delete time: %0.6f, delete size: %d\n", kdtree_delete_time, kdtree_delete_counter);
    // printf("Delete Box: %d\n",int(cub_needrm.size()));
}

#endif

//; 通用LiDAR类型的回调函数，比如机械式的LiDAR
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    // cout<<"got feature"<<endl;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    // ROS_INFO("get point cloud at time: %.6f and size: %d", msg->header.stamp.toSec() - 0.1, ptr->points.size());
    //    ROS_INFO("get point cloud at time: %.6f and size: %d", msg->header.stamp.toSec(), ptr->points.size());
    printf("[ INFO ]: get standard point cloud at time: %.6f and size: %d.\n", msg->header.stamp.toSec(),
           int(ptr->points.size()));
    lidar_buffer.push_back(ptr);
    // time_buffer.push_back(msg->header.stamp.toSec() - 0.1);
    // last_timestamp_lidar = msg->header.stamp.toSec() - 0.1;
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

//; livox激光雷达的消息回调函数
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    //    ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
    // printf("[ INFO ]: get livox point cloud at time: %.6f.\n", msg->header.stamp.toSec());
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    //; 对收到的原始点云进行一些预处理，得到后面要使用的面点
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

//; IMU消息的回调函数：存储IMU消息到buf中
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    //cout<<"msg_in:"<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

/**
 * @brief 从ros消息中把图像数据转成cv::Mat类型
 *  参考博客：https://blog.csdn.net/bigdog_1027/article/details/79090571
 *  https://zhuanlan.zhihu.com/p/392285419
 */
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv::Mat img;
    //; cv_bridge::toCvShare(img_msg, "bgr8") 是一个cv_bridge::CvImagePtr类型的指针，
    //; 这里用的是匿名对象的写法，然后直接调用这个指针的成员变量 image，就得到了对应的cv::Mat类型的数据
    img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
    //; 下面这个修改是github上一个人提的issue说的，说图片比较大的时候使用toCvShare容易出问题，
    //; 但是作者说测试过最大500w像素的图片没有出问题，因此没有采纳这个issue的建议
    // img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;

    return img;
}

//; 图像消息时间戳
void img_cbk(const sensor_msgs::ImageConstPtr &msg)
{
    if (!img_en)
    {
        return;
    }
    //    ROS_INFO("get img at time: %.6f", msg->header.stamp.toSec());
    // printf("[ INFO ]: get img at time: %.6f.\n", msg->header.stamp.toSec());
    if (msg->header.stamp.toSec() < last_timestamp_img)
    {
        ROS_ERROR("img loop back, clear buffer");
        img_buffer.clear();
        img_time_buffer.clear();
    }
    mtx_buffer.lock();
    // cout<<"Lidar_buff.size()"<<lidar_buffer.size()<<endl;
    // cout<<"Imu_buffer.size()"<<imu_buffer.size()<<endl;
    img_buffer.push_back(getImageFromMsg(msg));
    img_time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_img = msg->header.stamp.toSec();
    // cv::imshow("img", img);
    // cv::waitKey(1);
    // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


/**
 * @brief 以LiDAR时间戳为一次处理前提的数据包对齐。
 *          LiDAR时间戳：   |            |           |   （注意以一帧点云结尾时间戳为准）
 *                         1            2           3
 *           相机时间戳 ：      |   |   |   |   |   |   |
 *                            1   2   3   4   5   6   7
 * 两种情况：1. 当前最新帧就是LiDAR，没有更老的图像，比如上面第1帧lidar，那么同步的方法和LIO一样，非常简单
 *         2. 当前必须有一个LIDAR，比如上面第2帧lidar。但是LiDAR前面有一些更老的图像没有处理，比如
 *            第1、2、3帧图像，那么此时就要先同步这三帧图像。但是显然这种方法对图像的处理是有一些滞后的， 
 *            因为它要求必须有了一帧LiDAR才能对 比这帧LiDAR更老的图像进行处理。不过lidar是10hz，也不会
 *            滞后太多，因此问题也不大。
 * @param[in] meas 
 * @return true 
 * @return false 
 */
bool sync_packages(LidarMeasureGroup &meas)
{
    if ((lidar_buffer.empty() && img_buffer.empty()))
    { // has lidar topic or img topic?
        return false;
    }
    // ROS_ERROR("In sync");
    // If meas.is_lidar_end==true, means it just after scan end, clear all buffer in meas. 一次扫描结束
    //; 如果上次同步的消息是以lidar为结尾的，说明上次处理了一帧以lidar为结尾的数据，那么这次统计
    //; 图像数据的时候就要先清空了，因为measures里面存储的是以上一帧LiDAR为结尾的多帧的图像数据和IMU数据
    if (meas.is_lidar_end) 
    {
        meas.measures.clear();
        meas.is_lidar_end = false;
    }

    if (!lidar_pushed)  // If not in lidar scan, need to generate new meas
    {  
        //! 疑问：这里这种时间戳对齐要求当前对齐一帧图像或者LiDAR数据的时候，必须有LiDAR数据存在。
        // 也就是视觉的数据会被延迟最大0.1s处理。比如图像是30hz, LiDAR是10hz，他们都经过硬件时间同步，
        // 也就是时间戳是完全准确的，没有任何时间偏移。如下图所示：
        //                   1           2           3
        //    LiDAR时间戳：   |           |           | 
        //    相机时间戳：       |   |   |   |   |   |   |
        //                     1   2   3   4   5   6   7
        // (1)假设LiDAR先来，那么后面会先把第1帧LiDAR处理掉。
        // (2)第二次同步的时候进入当前if分支，发现LiDAR是空的，那么直接返回。此时即使有几帧相机数据，
        //    但是由于没有第2帧的LiDAR数据，所以这里仍然不会处理相机数据，而是要等到第2帧LiDAR数据
        //    来了之后才会往下进行。往下进行发现有一些比第2帧LiDAR数据更老的图像数据，也就是相机的
        //    前3帧都没有被处理，所以此时才会处理这3帧图像数据。处理完这3帧图像数据之后，然后继续
        //    处理第2帧的LiDAR数据。
        //; 另外注意：这里的lidar时间戳是以一帧的结束为标准的，因为后面去畸变是把一帧点云对齐到结尾
        if (lidar_buffer.empty())
        {
            // ROS_ERROR("out sync");
            return false;
        }
        //; 这里的LiDAR是单独存成了PCL点云格式，所以后面用一个单独的time_buffer来存储它的时间戳了
        meas.lidar = lidar_buffer.front(); // push the first lidar topic
        //; 如果这帧lidar点云无效，则要弹出图像数据。但是这个地方正常来说应该不会发生？
        if (meas.lidar->points.size() <= 1)
        {
            mtx_buffer.lock();
            // temp method, ignore img topic when no lidar points, keep sync
            if (img_buffer.size() > 0)
            {
                lidar_buffer.pop_front();
                img_buffer.pop_front();
            }
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            // ROS_ERROR("out sync");
            return false;
        }
        // sort by sample timestamp; small to big
        //; 对lidar中的点云根据时间戳进行排序
        sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list); 
        // generate lidar_beg_time // 雷达开始时间
        //! 疑问：这个地方感觉和前面 lidar_buffer.pop_front(); 不同步？但是正常来说前面的问题应该不会发生
        meas.lidar_beg_time = time_buffer.front();   
        //; 一帧lidar结束的绝对时间戳                          
        lidar_end_time =
            meas.lidar_beg_time +
            meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time 雷达扫描结束时间
        //; lidar_pushed 表示meas中的lidar点云插入了，但是还没有从buf中弹出
        lidar_pushed = true;                                    // flag
    }

    //; 如果图像为空，则只需要统计IMU消息
    if (img_buffer.empty())
    { 
        // no img topic, means only has lidar topic
        //; +0.02是为了稍微多要一点IMU数据，从而完整包括一帧LiDAR的数据
        if (last_timestamp_imu < lidar_end_time + 0.02)  
        { // imu message needs to be larger than lidar_end_time, keep complete propagate.
            // ROS_ERROR("out sync");
            return false;
        }
        struct MeasureGroup m; //standard method to keep imu message.
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();
        mtx_buffer.lock();
        while ((!imu_buffer.empty() && (imu_time < lidar_end_time)))
        { 
            // hr: make sure m.imu_end_time > lidar_end_time
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > lidar_end_time)
                break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
        //; 现在真正统计完一次以LiDAR为结尾的数据了，所以要把LiDAR消息和对应的时间戳弹出
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        //; lidar_pushed=true，说明 meas 插入了LiDAR消息，但是还没有同步完成，也就是在buffer中还有这个lidar消息
        //; 而如果=fasle，说明 meas 插入了LIDAR消息并且同步完成了，也就是buffer中已经弹出这个消息了
        lidar_pushed = false;     // sync one whole lidar scan.
        // process lidar topic, so timestamp should be lidar scan end.
        meas.is_lidar_end = true;  //; 这个表示当前对齐的一帧消息是否是以LiDAR的时间戳为结尾的
        meas.measures.push_back(m);
        // ROS_ERROR("out sync");
        return true;
    }
    
    //; 运行到这里，说明图像不为空，则要同时统计图像和LiDAR的时间戳
    struct MeasureGroup m;
    // cout<<"lidar_buffer.size(): "<<lidar_buffer.size()<<" img_buffer.size(): "<<img_buffer.size()<<endl;
    // cout<<"time_buffer.size(): "<<time_buffer.size()<<" img_time_buffer.size(): "<<img_time_buffer.size()<<endl;
    // cout<<"img_time_buffer.front(): "<<img_time_buffer.front()<<"lidar_end_time: "<<lidar_end_time<<"last_timestamp_imu: "<<last_timestamp_imu<<endl;
    //; 如果图像时间更晚，则当前仍然要处理lidar的帧，然后这里的操作就和前面图像时间戳为空的操作是一样的
    if ((img_time_buffer.front() > lidar_end_time))
    { // has img topic, but img topic timestamp larger than lidar end time, process lidar topic.
        if (last_timestamp_imu < lidar_end_time + 0.02)
        {
            // ROS_ERROR("out sync");
            return false;
        }
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();
        mtx_buffer.lock();
        while ((!imu_buffer.empty() && (imu_time < lidar_end_time)))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > lidar_end_time)
                break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        lidar_pushed = false;
        meas.is_lidar_end = true;
        meas.measures.push_back(m);
    }
    //; 否则图像时间更早，那么就要以图像的时间戳为结尾，统计IMU数据
    else
    {  
        // img topic timestamp smaller than lidar end time <=
        double img_start_time = img_time_buffer.front(); // process img topic, record timestamp
        if (last_timestamp_imu < img_start_time)
        {
            // ROS_ERROR("out sync");
            return false;
        }
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();
        // record img offset time, it shoule be the Kalman update timestamp.
        m.img_offset_time = img_start_time - meas.lidar_beg_time; 
        m.img = img_buffer.front();
        mtx_buffer.lock();
        while ((!imu_buffer.empty() && (imu_time < img_start_time)))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > img_start_time)
                break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        // has img topic in lidar scan, so flag "is_lidar_end=false"
        meas.is_lidar_end = false; 
        //; 这里可以发现没有对measures之前的图像数据清空，也就是当前帧LiDAR之前的多帧图像每次同步
        //; 都会被放到measures里面。后面看一下处理的时候怎么做的，肯定不会重复处理
        meas.measures.push_back(m);
    }
    // ROS_ERROR("out sync");
    return true;
}

void map_incremental()
{
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
    }
#ifdef USE_ikdtree
#ifdef USE_ikdforest
    ikdforest.Add_Points(feats_down_world->points, lidar_end_time);
#else
    ikdtree.Add_Points(feats_down_world->points, true);
#endif
#endif
}

// PointCloudXYZRGB::Ptr pcl_wait_pub_RGB(new PointCloudXYZRGB(500000, 1));
//; 在一次LIO优化后，当前LiDAR帧的点云转到世界坐标系下的点云
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());

void publish_frame_world_rgb(const ros::Publisher &pubLaserCloudFullRes, lidar_selection::LidarSelectorPtr lidar_selector)
{
    uint size = pcl_wait_pub->points.size();
    PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));
    if (img_en)
    {
        laserCloudWorldRGB->clear();
        for (int i = 0; i < size; i++)
        {
            PointTypeRGB pointRGB;
            pointRGB.x = pcl_wait_pub->points[i].x;
            pointRGB.y = pcl_wait_pub->points[i].y;
            pointRGB.z = pcl_wait_pub->points[i].z;
            V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
            V2D pc(lidar_selector->new_frame_->w2c(p_w));
            //; 把上一帧的LIDAR点云投影到当前帧的相机坐标系下，找对应的颜色给点云赋值
            if (lidar_selector->new_frame_->cam_->isInFrame(pc.cast<int>(), 0))
            {
                // cv::Mat img_cur = lidar_selector->new_frame_->img();
                cv::Mat img_rgb = lidar_selector->img_rgb;
                V3F pixel = lidar_selector->getpixel(img_rgb, pc);
                pointRGB.r = pixel[2]; // rgb信息
                pointRGB.g = pixel[1];
                pointRGB.b = pixel[0];
                laserCloudWorldRGB->push_back(pointRGB);
            }
        }
    }
    if (1) //if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        if (img_en)
        {
            // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
            pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
        }
        else
        {
            pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        }
        laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD; // publish_count以imu的发布为准 PUBFRAME_PERIOD：20
    }
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes)
{
    uint size = pcl_wait_pub->points.size();
    if (1) //if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;

        pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);

        laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
}

void publish_visual_world_map(const ros::Publisher &pubVisualCloud)
{
    PointCloudXYZI::Ptr laserCloudFullRes(map_cur_frame_point);
    int size = laserCloudFullRes->points.size();
    if (size == 0)
        return;
    PointCloudXYZI::Ptr pcl_visual_wait_pub(new PointCloudXYZI());
    *pcl_visual_wait_pub = *laserCloudFullRes;
    if (1) //if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*pcl_visual_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubVisualCloud.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
}

void publish_visual_world_sub_map(const ros::Publisher &pubSubVisualCloud)
{
    //; 当前帧图像用的很稀疏的视觉地图点云
    PointCloudXYZI::Ptr laserCloudFullRes(sub_map_cur_frame_point);
    int size = laserCloudFullRes->points.size();
    if (size == 0)
        return;
    PointCloudXYZI::Ptr sub_pcl_visual_wait_pub(new PointCloudXYZI());
    *sub_pcl_visual_wait_pub = *laserCloudFullRes;
    if (1) //if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*sub_pcl_visual_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubSubVisualCloud.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time::now();
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    out.position.x = state.pos_end(0);
    out.position.y = state.pos_end(1);
    out.position.z = state.pos_end(2);
    out.orientation.x = geoQuat.x;
    out.orientation.y = geoQuat.y;
    out.orientation.z = geoQuat.z;
    out.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
    set_posestamp(odomAftMapped.pose.pose);
    pubOdomAftMapped.publish(odomAftMapped);
}

void publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_odom_frame";
    set_posestamp(msg_body_pose.pose);
    mavros_pose_publisher.publish(msg_body_pose);
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_init";
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
}

//; 从ROS参数服务器中读取参数
void readParameters(ros::NodeHandle &nh)
{
    nh.param<int>("dense_map_enable", dense_map_en, 1);
    nh.param<int>("img_enable", img_en, 1);
    nh.param<int>("lidar_enable", lidar_en, 1);
    nh.param<int>("debug", debug, 0);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<bool>("ncc_en", ncc_en, false);
    nh.param<int>("min_img_count", MIN_IMG_COUNT, 1000);
    nh.param<double>("cam_fx", cam_fx, 453.483063); // 相机内参
    nh.param<double>("cam_fy", cam_fy, 453.254913);
    nh.param<double>("cam_cx", cam_cx, 318.908851);
    nh.param<double>("cam_cy", cam_cy, 234.238189);
    nh.param<double>("laser_point_cov", LASER_POINT_COV, 0.001);
    nh.param<double>("img_point_cov", IMG_POINT_COV, 10);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<string>("camera/img_topic", img_topic, "/usb_cam/image_raw");
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<double>("mapping/fov_degree", fov_deg, 180); // FOV
    nh.param<double>("mapping/gyr_cov_scale", gyr_cov_scale, 1.0);
    nh.param<double>("mapping/acc_cov_scale", acc_cov_scale, 1.0);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, 0);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 雷达imu外参
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<vector<double>>("camera/Pcl", cameraextrinT, vector<double>()); // 相机雷达外参
    nh.param<vector<double>>("camera/Rcl", cameraextrinR, vector<double>());
    nh.param<int>("grid_size", grid_size, 40);    // 每个网格的像素宽高，配置为 40
    nh.param<int>("patch_size", patch_size, 4);   // 选择的 patch 的宽高，配置为 8
    nh.param<double>("outlier_threshold", outlier_threshold, 100);
    nh.param<double>("ncc_thre", ncc_thre, 100);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh); // ros 用于image订阅和发布
    readParameters(nh);
    cout << "debug:" << debug << " MIN_IMG_COUNT: " << MIN_IMG_COUNT << endl;
    pcl_wait_pub->clear(); // TODO：等待发布的点云？ note：world frame
    //; 订阅LiDAR、IMU、img消息
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? 
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_img = nh.subscribe(img_topic, 200000, img_cbk);
    image_transport::Publisher img_pub = it.advertise("/rgb_img", 1);
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
    ros::Publisher pubLaserCloudFullResRgb = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_rgb", 100);
    ros::Publisher pubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_map", 100);
    ros::Publisher pubSubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map", 100);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 10);

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** variables definition ***/
    VD(DIM_STATE) solution; // 18*1
    MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE; // 18*18
    V3D rot_add, t_add;
    StatesGroup state_propagat;
    PointType pointOri, pointSel, coeff;

    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0); // TODO：没用到，传入fov_deg有什么用？

    // 降采样系数
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    //; IMU处理的函数
    shared_ptr<ImuProcess> p_imu(new ImuProcess());

    //; LiDAR和IMu的外参
    V3D extT;
    M3D extR;
    extT << VEC_FROM_ARRAY(extrinT);
    extR << MAT_FROM_ARRAY(extrinR);
    Lidar_offset_to_IMU = extT;

    //! 重要：处理VIO部分的类
    lidar_selection::LidarSelectorPtr lidar_selector(
        new lidar_selection::LidarSelector(grid_size, new SparseMap));
    //; 从命名空间中读取参数，生成一个虚拟的相机类
    if (!vk::camera_loader::loadFromRosNs("laserMapping", lidar_selector->cam))
        throw std::runtime_error("Camera model not correctly specified.");
    // TODO：初始化lidar_selection的一些参数
    //; 这个没用到
    lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT;   // 1000
    lidar_selector->debug = debug;  // 0 是否显示debug信息
    lidar_selector->patch_size = patch_size;   // 8
    lidar_selector->outlier_threshold = outlier_threshold;  // 300
    lidar_selector->ncc_thre = ncc_thre;   // 0 ncc 的阈值
    //; 进去内部看，应该是 T_camera_lidar?
    lidar_selector->sparse_map->set_camera2lidar(cameraextrinR, cameraextrinT); // hr: from camera to lidar
    //; 传入的是 T_imu_lidar，内部赋值做了转换，变成 T_lidar_imu
    lidar_selector->set_extrinsic(extT, extR);  // hr: TODO:return T from imu to lidar
    //; 绑定状态变量，这样会在VIO里面直接更改LIO的结果
    lidar_selector->state = &state;  
    //; IMU预测的状态，这和IEKF有关，因为IEKF会一直计算当前状态和预测状态之间的差值
    lidar_selector->state_propagat = &state_propagat;  
    lidar_selector->NUM_MAX_ITERATIONS = NUM_MAX_ITERATIONS; // 4，IEKF迭代的最大阈值
    //; 和优化有关，视觉点的协方差
    lidar_selector->img_point_cov = IMG_POINT_COV; // 100
    //; 给成员变量中的相机内参赋值
    lidar_selector->fx = cam_fx;
    lidar_selector->fy = cam_fy;
    lidar_selector->cx = cam_cx;
    lidar_selector->cy = cam_cy;
    //; NCC是归一化相关性，是相比使用patch对齐的更复杂的差异度量方式，见十四讲P230
    lidar_selector->ncc_en = ncc_en; // 0
    lidar_selector->init();
    //------------------------------- vio 部分变量初始化完毕 --------------------------


    //; 对IMU类设置噪声等消息
    p_imu->set_extrinsic(extT, extR); // TODO:lidar to imu??
    p_imu->set_gyr_cov_scale(V3D(gyr_cov_scale, gyr_cov_scale, gyr_cov_scale));
    p_imu->set_acc_cov_scale(V3D(acc_cov_scale, acc_cov_scale, acc_cov_scale));
    //    p_imu->set_gyr_bias_cov(V3D(0.00001, 0.00001, 0.00001));
    //    p_imu->set_acc_bias_cov(V3D(0.00001, 0.00001, 0.00001));
    p_imu->set_gyr_bias_cov(V3D(0.00003, 0.00003, 0.00003));
    p_imu->set_acc_bias_cov(V3D(0.01, 0.01, 0.01));

    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);

    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        // Step 1: 同步LiDAR、IMU、Image信息，如果没有同步成功，则一直在这里等待
        if (!sync_packages(LidarMeasures))
        {
            status = ros::ok();
            cv::waitKey(1);
            rate.sleep();
            continue;
        }

        /*** Packaged got ***/
        if (flg_reset)
        {
            ROS_WARN("reset when rosbag play back");
            p_imu->Reset();
            flg_reset = false;
            continue;
        }

        // double t0,t1,t2,t3,t4,t5,match_start, match_time, solve_start, solve_time, svd_time;
        double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

        match_time = kdtree_search_time = kdtree_search_counter = solve_time = solve_const_H_time = svd_time = 0;
        t0 = omp_get_wtime();

        double time_start = t0;

        // Step 2: 利用IMU数据对状态变量进行积分递推，同时得到去畸变之后的LIDAR点云
        //! 疑问：里面的代码太乱，没有看懂如果当前帧是图像，到底有没有对点云进行去畸变
        //! 暂时解答：感觉应该是没有去畸变处理的，以为里面的操作如果是图像则点的时间都不满足要求，都不会去畸变
        p_imu->Process2(LidarMeasures, state, feats_undistort);
        state_propagat = state;

        if (lidar_selector->debug)
        {
            LidarMeasures.debug_show();
        }

        if (feats_undistort->empty() || (feats_undistort == nullptr))
        {
            cout << " No point!!!" << endl;
            if (!fast_lio_is_ready)
            {
                first_lidar_time = LidarMeasures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                LidarMeasures.measures.clear();
                cout << "FAST-LIO not ready" << endl;
                continue;
            }
        }
        else
        {
            int size = feats_undistort->points.size();
        }
        fast_lio_is_ready = true;
        // <0.5 false; >=0.5 true
        flg_EKF_inited = (LidarMeasures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true; 
        
        // Step 3: 当前是图像帧，则进行视觉VIO处理
        if (!LidarMeasures.is_lidar_end)
        {
            //            cout << "[ VIO ]: Raw feature num: " << feats_undistort->points.size() << endl;
            //; 打印本次VIO之前，上一次LIO的LiDAR点云(转到world系下)
            // cout << "[ VIO ]: Raw feature num: " << pcl_wait_pub->points.size() << "." << endl;
            if (first_lidar_time < 10)
            { 
                // TODO: threshold lidar_begin_time
                continue;
            }
            //; 如果开启VIO模块，则才往下处理
            if (img_en)
            {
                euler_cur = RotMtoEuler(state.rot_end);
                fout_pre << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " "
                         << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " "
                         << state.vel_end.transpose() << " " << state.bias_g.transpose() << " "
                         << state.bias_a.transpose() << " " << state.gravity.transpose() << endl;

                /* visual main */
                //! 重要：视觉VIO的主函数！
                //; 传入: 当前帧的图像 和 上一帧的LiDAR在世界坐标系下的点云
                lidar_selector->detect(LidarMeasures.measures.back().img, pcl_wait_pub); 

                double time_end = omp_get_wtime();
                std::cout << "-- vio time: " << (time_end - time_start) << std::endl;

                // int size = lidar_selector->map_cur_frame_.size();
                int size_sub = lidar_selector->sub_map_cur_frame_.size();

                // map_cur_frame_point->clear();
                sub_map_cur_frame_point->clear();

                for (int i = 0; i < size_sub; i++)
                {
                    PointType temp_map;
                    temp_map.x = lidar_selector->sub_map_cur_frame_[i]->pos_[0];
                    temp_map.y = lidar_selector->sub_map_cur_frame_[i]->pos_[1];
                    temp_map.z = lidar_selector->sub_map_cur_frame_[i]->pos_[2];
                    temp_map.intensity = 0.;
                    sub_map_cur_frame_point->push_back(temp_map);
                }
                // cout<<"2222222222222";
                // cout<<"new_frame_: "<<lidar_selector->new_frame_->id_<<endl;
                cv::Mat img_rgb = lidar_selector->img_cp;
                cv_bridge::CvImage out_msg;
                out_msg.header.stamp = ros::Time::now();
                // out_msg.header.frame_id = "camera_init";
                out_msg.encoding = sensor_msgs::image_encodings::BGR8;
                out_msg.image = img_rgb;
                img_pub.publish(out_msg.toImageMsg());

                // 发布带有rgb信息的点云信息
                publish_frame_world_rgb(pubLaserCloudFullResRgb, lidar_selector); 
                // 发布sub_map_cur_frame_point
                publish_visual_world_sub_map(pubSubVisualCloud);  

                geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
                publish_odometry(pubOdomAftMapped);
                euler_cur = RotMtoEuler(state.rot_end);
                fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " "
                         << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " "
                         << state.vel_end.transpose()
                         << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << " "
                         << feats_undistort->points.size() << endl;
            }
            //; 不用继续往下处理了，因为当前只是处理视觉的部分，而不包括激光的信息
            continue;
        }

        // Step 4: 运行到这里，说明当前是LiDAR帧，则运行LIO
        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();

        /*** downsample the feature points in a scan ***/
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);

        /*** initialize the map kdtree ***/
        if (ikdtree.Root_Node == nullptr)
        {
            if (feats_down_body->points.size() > 5)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                ikdtree.Build(feats_down_body->points);
            }
            continue;
        }
        int featsFromMapNum = ikdtree.size();

        feats_down_size = feats_down_body->points.size();
        // cout << "[ LIO ]: Raw feature num: " << feats_undistort->points.size() << " downsamp num " << feats_down_size
        //      << " Map num: " << featsFromMapNum << "." << endl;

        /*** ICP and iterated Kalman filter update ***/
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        //vector<double> res_last(feats_down_size, 1000.0); // initial //
        res_last.resize(feats_down_size, 1000.0);

        t1 = omp_get_wtime();
        if (lidar_en)
        {
            euler_cur = RotMtoEuler(state.rot_end);
            fout_pre << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " "
                     << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " "
                     << state.vel_end.transpose()
                     << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << endl;
        }

        if (0)
        {
            PointVector().swap(ikdtree.PCL_Storage);
            ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
            featsFromMap->clear();
            featsFromMap->points = ikdtree.PCL_Storage;
        }

        point_selected_surf.resize(feats_down_size, true);
        pointSearchInd_surf.resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        int rematch_num = 0;
        bool nearest_search_en = true; //

        t2 = omp_get_wtime();

        /*** iterated state estimation ***/
        double t_update_start = omp_get_wtime();

        if (img_en)
        {
            omp_set_num_threads(MP_PROC_NUM); // 设置线程的默认周期数为MP_PROC_NUM 4
#pragma omp parallel for // 并行计算
            for (int i = 0; i < 1; i++)
            {
            }
        }

        if (lidar_en)
        {
            for (iterCount = -1; iterCount < NUM_MAX_ITERATIONS && flg_EKF_inited; iterCount++)
            {
                match_start = omp_get_wtime();
                PointCloudXYZI().swap(*laserCloudOri);
                PointCloudXYZI().swap(*corr_normvect);
                // laserCloudOri->clear();
                // corr_normvect->clear();
                total_residual = 0.0;

                /** closest surface search and residual computation **/
                for (int i = 0; i < feats_down_size; i++)
                {
                    PointType &point_body = feats_down_body->points[i];
                    PointType &point_world = feats_down_world->points[i];
                    V3D p_body(point_body.x, point_body.y, point_body.z);
                    /* transform to world frame */
                    pointBodyToWorld(&point_body, &point_world);
                    vector<float> pointSearchSqDis(NUM_MATCH_POINTS); // #define 5

                    auto &points_near = Nearest_Points[i];
                    uint8_t search_flag = 0;
                    double search_start = omp_get_wtime();
                    if (nearest_search_en)
                    {
                        /** Find the closest surfaces in the map **/
                        ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                        point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
                        kdtree_search_time += omp_get_wtime() - search_start;
                        kdtree_search_counter++;
                    }

                    if (!point_selected_surf[i] || points_near.size() < NUM_MATCH_POINTS)
                        continue;

                    VF(4)
                    pabcd; // Matrix<float, (4), 1>
                    point_selected_surf[i] = false;
                    if (esti_plane(pabcd, points_near, 0.1f)) //(planeValid)
                    {
                        float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z +
                                    pabcd(3);
                        float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm()); // fabs(pd2) / sqrt(p_body.norm()):点到平面距离公式

                        //                        normvec->resize(feats_down_size);
                        if (s > 0.9)
                        { // TODO: threshold 点到平面距离 < 1/9
                            point_selected_surf[i] = true;
                            normvec->points[i].x = pabcd(0); // hr: save normvec
                            normvec->points[i].y = pabcd(1);
                            normvec->points[i].z = pabcd(2);
                            normvec->points[i].intensity = pd2;
                            res_last[i] = abs(pd2); // hr: save residuals
                        }
                    }
                }
                // cout<<"pca time test: "<<pca_time1<<" "<<pca_time2<<endl;
                effct_feat_num = 0;
                laserCloudOri->resize(feats_down_size);
                corr_normvect->reserve(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    if (point_selected_surf[i] && (res_last[i] <= 2.0))
                    { // TODO:threshold residuals <= 2.
                        laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                        corr_normvect->points[effct_feat_num] = normvec->points[i];
                        total_residual += res_last[i];
                        effct_feat_num++;
                    }
                }

                res_mean_last = total_residual / effct_feat_num;
                // debug:
                // cout << "[ mapping ]: Effective feature num: " << effct_feat_num << " res_mean_last " << res_mean_last
                //      << endl;
                // printf("[ LIO ]: time: fov_check: %0.6f fov_check and readd: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f icp: %0.6f construct H: %0.6f.\n",
                //        fov_check_time, t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu,
                //        aver_time_icp, aver_time_const_H_time);
                match_time += omp_get_wtime() - match_start; // 匹配结束
                solve_start = omp_get_wtime();               // 迭代求解开始

                /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
                MatrixXd Hsub(effct_feat_num, 6);
                VectorXd meas_vec(effct_feat_num);

                for (int i = 0; i < effct_feat_num; i++)
                {
                    const PointType &laser_p = laserCloudOri->points[i];
                    V3D point_this(laser_p.x, laser_p.y, laser_p.z);
                    point_this += Lidar_offset_to_IMU;
                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this); // 斜对称矩阵

                    /*** get the normal vector of closest surface/corner ***/
                    const PointType &norm_p = corr_normvect->points[i];
                    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

                    /*** calculate the Measuremnt Jacobian matrix H ***/
                    V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
                    Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;

                    /*** Measuremnt: distance to the closest surface/corner ***/
                    meas_vec(i) = -norm_p.intensity;
                }
                solve_const_H_time += omp_get_wtime() - solve_start;

                MatrixXd K(DIM_STATE, effct_feat_num);

                EKF_stop_flg = false;
                flg_EKF_converged = false;

                /*** Iterative Kalman Filter Update ***/
                if (!flg_EKF_inited)
                {
                    cout << "||||||||||Initiallizing LiDar||||||||||" << endl;
                    /*** only run in initialization period ***/
                    MatrixXd H_init(MD(9, DIM_STATE)::Zero());
                    MatrixXd z_init(VD(9)::Zero());
                    H_init.block<3, 3>(0, 0) = M3D::Identity();
                    H_init.block<3, 3>(3, 3) = M3D::Identity();
                    H_init.block<3, 3>(6, 15) = M3D::Identity();
                    z_init.block<3, 1>(0, 0) = -Log(state.rot_end);
                    z_init.block<3, 1>(0, 0) = -state.pos_end;

                    auto H_init_T = H_init.transpose();
                    auto &&K_init = state.cov * H_init_T * (H_init * state.cov * H_init_T + 0.0001 * MD(9, 9)::Identity()).inverse();
                    solution = K_init * z_init;
                    state.resetpose(); // R：单位阵；p,v:Zero3d
                    EKF_stop_flg = true;
                }
                else
                {
                    auto &&Hsub_T = Hsub.transpose();
                    auto &&HTz = Hsub_T * meas_vec;
                    H_T_H.block<6, 6>(0, 0) = Hsub_T * Hsub;
                    // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
                    // TODO:雷达协方差
                    MD(DIM_STATE, DIM_STATE) &&K_1 =
                        (H_T_H + (state.cov / LASER_POINT_COV).inverse()).inverse();
                    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
                    auto vec = state_propagat - state;
                    solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec -
                               G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);

                    //                    int minRow, minCol;
                    if (0) //if(V.minCoeff(&minRow, &minCol) < 1.0f)
                    {
                        VD(6)
                        V = H_T_H.block<6, 6>(0, 0).eigenvalues().real();
                        cout << "!!!!!! Degeneration Happend, eigen values: " << V.transpose() << endl;
                        EKF_stop_flg = true;
                        solution.block<6, 1>(9, 0).setZero();
                    }

                    state += solution;

                    rot_add = solution.block<3, 1>(0, 0);
                    t_add = solution.block<3, 1>(3, 0);

                    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015))
                    { // TODO: threshold EKF收敛
                        flg_EKF_converged = true;
                    }

                    deltaR = rot_add.norm() * 57.3; // 弧度 to 角度
                    deltaT = t_add.norm() * 100;    // m to cm
                }
                euler_cur = RotMtoEuler(state.rot_end);

                /*** Rematch Judgement ***/
                nearest_search_en = false;
                if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2))))
                {
                    nearest_search_en = true;
                    rematch_num++;
                }

                /*** Convergence Judgements and Covariance Update ***/
                if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1)))
                {
                    if (flg_EKF_inited)
                    {
                        /*** Covariance Update ***/
                        // G.setZero();
                        // G.block<DIM_STATE,6>(0,0) = K * Hsub;
                        state.cov = (I_STATE - G) * state.cov;
                        total_distance += (state.pos_end - position_last).norm();
                        position_last = state.pos_end;
                        geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));

                        VD(DIM_STATE)
                        K_sum = K.rowwise().sum(); // Matrix<double, ((18)), 1>
                        VD(DIM_STATE)
                        P_diag = state.cov.diagonal();
                    }
                    EKF_stop_flg = true;
                }
                solve_time += omp_get_wtime() - solve_start;

                if (EKF_stop_flg)
                    break;
            }
        }

        // debug
        // cout << "[ mapping ]: iteration count: " << iterCount + 1 << endl;

        // SaveTrajTUM(LidarMeasures.lidar_beg_time, state.rot_end, state.pos_end);
        double t_update_end = omp_get_wtime();

        double time_end = t_update_end;
        std::cout << "LIO time: " << (time_end - time_start) << std::endl;

        /******* Publish odometry *******/
        euler_cur = RotMtoEuler(state.rot_end);
        geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
        publish_odometry(pubOdomAftMapped);

        /*** add the feature points to map kdtree ***/
        t3 = omp_get_wtime();
        map_incremental();
        t5 = omp_get_wtime();
        kdtree_incremental_time = t5 - t3 + readd_time;
        /******* Publish points *******/

        PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }
        *pcl_wait_pub = *laserCloudWorld;

        publish_frame_world(pubLaserCloudFullRes);
        // publish_visual_world_map(pubVisualCloud);
        publish_effect_world(pubLaserCloudEffect);
        // publish_map(pubLaserCloudMap);
        publish_path(pubPath);

        /*** Debug variables ***/
        frame_num++;
        aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
        aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
        aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;

        aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
        //cout << "construct H:" << aver_time_const_H_time << std::endl;
        // aver_time_consu = aver_time_consu * 0.9 + (t5 - t0) * 0.1;
        T1[time_log_counter] = LidarMeasures.lidar_beg_time;
        s_plot[time_log_counter] = aver_time_consu;
        s_plot2[time_log_counter] = kdtree_incremental_time;
        s_plot3[time_log_counter] = kdtree_search_time / kdtree_search_counter;
        s_plot4[time_log_counter] = featsFromMapNum;
        s_plot5[time_log_counter] = t5 - t0;
        time_log_counter++;
        // cout<<"[ mapping ]: time: fov_check "<< fov_check_time <<" fov_check and readd: "<<t1-t0<<" match "<<aver_time_match<<" solve "<<aver_time_solve<<" ICP "<<t3-t1<<" map incre "<<t5-t3<<" total "<<aver_time_consu << "icp:" << aver_time_icp << "construct H:" << aver_time_const_H_time <<endl;
        // printf("[ mapping ]: time: fov_check %0.6f fov_check and readd: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f icp: %0.6f construct H: %0.6f \n",
        //        fov_check_time, t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu,
        //        aver_time_icp, aver_time_const_H_time);
        if (lidar_en)
        {
            euler_cur = RotMtoEuler(state.rot_end);

            fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " "
                     << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " "
                     << state.vel_end.transpose()
                     << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << " "
                     << feats_undistort->points.size() << endl;
        }
        // dump_lio_state_to_log(fp);
    }
    //--------------------------save map---------------
    // string surf_filename(map_file_path + "/surf.pcd");
    // string corner_filename(map_file_path + "/corner.pcd");
    // string all_points_filename(map_file_path + "/all_points.pcd");

    // PointCloudXYZI surf_points, corner_points;
    // surf_points = *featsFromMap;
    // fout_out.close();
    // fout_pre.close();
    // if (surf_points.size() > 0 && corner_points.size() > 0)
    // {
    // pcl::PCDWriter pcd_writer;
    // cout << "saving...";
    // pcd_writer.writeBinary(surf_filename, surf_points);
    // pcd_writer.writeBinary(corner_filename, corner_points);
    // }
    vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
    FILE *fp2;
    string log_dir = root_dir + "/Log/fast_livo_time_log.csv";
    fp2 = fopen(log_dir.c_str(), "w");
    fprintf(fp2,
            "time_stamp, average time, incremental time, search time,fov check time, total time, alpha_bal, alpha_del\n");
    for (int i = 0; i < time_log_counter; i++)
    {
        fprintf(fp2, "%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%f,%f\n", T1[i], s_plot[i], s_plot2[i], s_plot3[i],
                s_plot4[i], s_plot5[i], s_plot6[i], s_plot7[i]);
        t.push_back(T1[i]);
        s_vec.push_back(s_plot[i]);
        s_vec2.push_back(s_plot2[i]);
        s_vec3.push_back(s_plot3[i]);
        s_vec4.push_back(s_plot4[i]);
        s_vec5.push_back(s_plot5[i]);
        s_vec6.push_back(s_plot6[i]);
        s_vec7.push_back(s_plot7[i]);
    }
    fclose(fp2);
    if (!t.empty())
    {
        // plt::named_plot("incremental time",t,s_vec2);
        // plt::named_plot("search_time",t,s_vec3);
        // plt::named_plot("total time",t,s_vec5);
        // plt::named_plot("average time",t,s_vec);
        // plt::legend();
        // plt::show();
        // plt::pause(0.5);
        // plt::close();
    }
    cout << "no points saved" << endl;

    return 0;
}
