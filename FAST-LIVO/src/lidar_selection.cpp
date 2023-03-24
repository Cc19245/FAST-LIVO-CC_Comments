#include "lidar_selection.h"

namespace lidar_selection
{
    LidarSelector::LidarSelector(const int gridsize, SparseMap *sparsemap) : grid_size(gridsize),
                                                                             sparse_map(sparsemap)
    {
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
        H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
        Rli = M3D::Identity();
        Rci = M3D::Identity();
        Rcw = M3D::Identity();
        Jdphi_dR = M3D::Identity();
        Jdp_dt = M3D::Identity();
        Jdp_dR = M3D::Identity();
        Pli = V3D::Zero();
        Pci = V3D::Zero();
        Pcw = V3D::Zero();
        width = 800;
        height = 600;
    }

    LidarSelector::~LidarSelector()
    {
        delete sparse_map;
        delete sub_sparse_map;
        delete[] grid_num;
        delete[] map_index;
        delete[] map_value;
        delete[] align_flag;
        delete[] patch_cache;
        unordered_map<int, Warp *>().swap(Warp_map);
        unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
        unordered_map<VOXEL_KEY, VOXEL_POINTS *>().swap(feat_map);
    }

    void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
    {
        Pli = -rot.transpose() * transl; // TODO:lidar to imu
        Rli = rot.transpose();
    }

    void LidarSelector::init()
    {
        sub_sparse_map = new SubSparseMap;
        Rci = sparse_map->Rcl * Rli;
        Pci = sparse_map->Rcl * Pli + sparse_map->Pcl;
        M3D Ric;
        V3D Pic;
        Jdphi_dR = Rci;
        Pic = -Rci.transpose() * Pci;
        M3D tmp;
        tmp << SKEW_SYM_MATRX(Pic); // 0.0, -Pic[2], Pic[1], Pic[2], 0.0, -Pic[0], -Pic[1], Pic[0], 0.0
        Jdp_dR = -Rci * tmp;
        width = cam->width();
        height = cam->height();
        grid_n_width = static_cast<int>(width / grid_size);   // 848/40
        grid_n_height = static_cast<int>(height / grid_size); // 480/40
        length = grid_n_width * grid_n_height;   //; 划分成grid的总数
        //! 疑问：这里为什么要这么操作？
        fx = cam->errorMultiplier2();  //; abs(fx_)
        fy = cam->errorMultiplier() / (4. * fx);  //; errorMultiplier返回是4*fx*fy
        grid_num = new int[length];

        map_index = new int[length];
        map_value = new float[length];
        align_flag = new int[length];
        map_dist = (float *)malloc(sizeof(float) * length);
        memset(grid_num, TYPE_UNKNOWN, sizeof(int) * length);
        memset(map_index, 0, sizeof(int) * length);
        memset(map_value, 0, sizeof(float) * length);

        voxel_points_.reserve(length);
        add_voxel_points_.reserve(length);
        count_img = 0;
        patch_size_total = patch_size * patch_size;
        patch_size_half = static_cast<int>(patch_size / 2);
        patch_cache = new float[patch_size_total];
        stage_ = STAGE_FIRST_FRAME;
        pg_down.reset(new PointCloudXYZI());
        Map_points.reset(new PointCloudXYZI());
        Map_points_output.reset(new PointCloudXYZI());
        weight_scale_ = 10;
        weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
        // weight_function_.reset(new vk::robust_cost::TukeyWeightFunction());
        scale_estimator_.reset(new vk::robust_cost::UnitScaleEstimator());
        // scale_estimator_.reset(new vk::robust_cost::MADScaleEstimator());
    }

    void LidarSelector::reset_grid()
    {
        memset(grid_num, TYPE_UNKNOWN, sizeof(int) * length); // length = grid_n_width * grid_n_height = 768
        //    cout << "111111111111111111: " << sizeof(int)*length << endl;
        memset(map_index, 0, sizeof(int) * length);
        fill_n(map_dist, length, 10000);
        //; 这里还是用swap函数，这样相当于对vector里面的内容进行清空，但是vector占用的
        //; 仍然是原来的内存，并不需要重新分配内存
        // 此时 voxel_points_ 和 add_voxel_points 是全新的，没有存任何数据
        std::vector<PointPtr>(length).swap(voxel_points_);  //; 本次找到的视觉地图中的点
        std::vector<V3D>(length).swap(add_voxel_points_);   //; 当前帧往视觉地图中新加入的点
        voxel_points_.reserve(length);     //; 网格中存储的3D点，注意只有一个，就是深度最近的那个点
        add_voxel_points_.reserve(length);
    }

    //; 像素点对相机系下的点的导数，十四讲P220公式(8.16)
    void LidarSelector::dpi(V3D p, MD(2, 3) & J)
    {
        const double x = p[0];
        const double y = p[1];
        const double z_inv = 1. / p[2];
        const double z_inv_2 = z_inv * z_inv;
        J(0, 0) = fx * z_inv;
        J(0, 1) = 0.0;
        J(0, 2) = -fx * x * z_inv_2;
        J(1, 0) = 0.0;
        J(1, 1) = fy * z_inv;
        J(1, 2) = -fy * y * z_inv_2;
    }

    float LidarSelector::CheckGoodPoints(cv::Mat img, V2D uv)
    {
        const float u_ref = uv[0];
        const float v_ref = uv[1];
        const int u_ref_i = floorf(uv[0]);
        const int v_ref_i = floorf(uv[1]);
        const float subpix_u_ref = u_ref - u_ref_i;
        const float subpix_v_ref = v_ref - v_ref_i;
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i)*width + (u_ref_i);
        float gu = 2 * (img_ptr[1] - img_ptr[-1]) + img_ptr[1 - width] - img_ptr[-1 - width] + img_ptr[1 + width] -
                   img_ptr[-1 + width];
        float gv =
            2 * (img_ptr[width] - img_ptr[-width]) + img_ptr[width + 1] - img_ptr[-width + 1] + img_ptr[width - 1] -
            img_ptr[-width - 1];
        return fabs(gu) + fabs(gv);
    }

    void LidarSelector::getpatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
    {
        const float u_ref = pc[0];
        const float v_ref = pc[1];
        const int scale = (1 << level);
        const int u_ref_i = floorf(pc[0] / scale) * scale; // 向下取整
        const int v_ref_i = floorf(pc[1] / scale) * scale;
        const float subpix_u_ref = (u_ref - u_ref_i) / scale;
        const float subpix_v_ref = (v_ref - v_ref_i) / scale;
        const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
        const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
        const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
        const float w_ref_br = subpix_u_ref * subpix_v_ref;
        for (int x = 0; x < patch_size; x++)
        {
            uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width +
                               (u_ref_i - patch_size_half * scale);
            for (int y = 0; y < patch_size; y++, img_ptr += scale)
            {
                patch_tmp[patch_size_total * level + x * patch_size + y] =
                    w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                    w_ref_br * img_ptr[scale * width + scale]; // TODO:??
            }
        }
    }

    /**
     * @brief 往地图中新加入地图点
     * 
     * @param[in] img 
     * @param[in] pg 
     */
    void LidarSelector::addSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg)
    {
        double t0 = omp_get_wtime();
        reset_grid();

        double t_b1 = omp_get_wtime() - t0;
        t0 = omp_get_wtime();

        // Step 1: 把上一帧的LiDAR点云投影到当前帧的图像上，计算这个点的角点得分，如果超过地图点得分，则新加入这个点
        for (int i = 0; i < pg->size(); i++)
        {
            V3D pt(pg->points[i].x, pg->points[i].y, pg->points[i].z);
            V2D pc(new_frame_->w2c(pt));
            if (new_frame_->cam_->isInFrame(pc.cast<int>(),
                                            (patch_size_half + 1) * 8)) // 20px is the patch size in the matcher
            {
                int index = static_cast<int>(pc[0] / grid_size) * grid_n_height + static_cast<int>(pc[1] / grid_size);
                // float cur_value = CheckGoodPoints(img, pc);

                //; 计算这个lidar点对应的视觉特征点的 shiTomasi 得分，得分超过原先地图中的patch得分的话，加入这个点的观测
                float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]); // 计算角点评分，得分越高，则特征越优先

                //! only add in not occupied grid
                if (cur_value > map_value[index]) //&& (grid_num[index] != TYPE_MAP || map_value[index]<=10)) 
                {
                    map_value[index] = cur_value;
                    add_voxel_points_[index] = pt;  //; 注意这里没有覆盖地图点，而是存到了一个新的变量里面
                    grid_num[index] = TYPE_POINTCLOUD;
                }
            }
        }

        double t_b2 = omp_get_wtime() - t0;
        t0 = omp_get_wtime();

        // Step 2: 再次遍历所有网格，看是否有新加入的点
        int add = 0;
        for (int i = 0; i < length; i++)
        {
            if (grid_num[i] == TYPE_POINTCLOUD) // && (map_value[i]>=10)) //! debug
            {
                V3D pt = add_voxel_points_[i];
                V2D pc(new_frame_->w2c(pt));
                float *patch = new float[patch_size_total * 3];
                //; 获取三个不同尺度的patch
                getpatch(img, pc, patch, 0);
                getpatch(img, pc, patch, 1);
                getpatch(img, pc, patch, 2);
                PointPtr pt_new(new Point(pt));
                Vector3d f = cam->cam2world(pc); // Project from pixels to world coordiantes. Returns a bearing vector of unit length.
                FeaturePtr ftr_new(new Feature(patch, pc, f, new_frame_->T_f_w_, map_value[i], 0));
                ftr_new->img = new_frame_->img_pyr_[0];
                // ftr_new->ImgPyr.resize(5);
                // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
                ftr_new->id_ = new_frame_->id_;

                pt_new->addFrameRef(ftr_new);
                pt_new->value = map_value[i];
                AddPoint(pt_new);
                add += 1;
            }
        }

        double t_b3 = omp_get_wtime() - t0;

        //        printf("[ Add ] : %d 3D points \n", add);
        // printf("pg.size: %d \n", pg->size());  //; 这一帧点云总数
        //        printf("B1. : %.6lf \n", t_b1);
        //        printf("B2. : %.6lf \n", t_b2);
        //        printf("B3. : %.6lf \n", t_b3);
        // printf("[ VIO ]: Add %d 3D points.\n", add);  //; 这一帧新加入的3D点
    }

    void LidarSelector::AddPoint(PointPtr pt_new)
    {
        V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
        double voxel_size = 0.5;
        float loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = pt_w[j] / voxel_size;
            if (loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_KEY position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end())
        {
            iter->second->voxel_points.push_back(pt_new);
            iter->second->count++;
        }
        else
        {
            VOXEL_POINTS *ot = new VOXEL_POINTS(0);
            ot->voxel_points.push_back(pt_new);
            feat_map[position] = ot;
        }
    }

    /**
     * @brief 输入当前帧相机的一个像素坐标，和当前帧与另一帧之间的位姿变换，计算由于这个位姿变换导致的像素的affine变换
     * 
     */
    void LidarSelector::getWarpMatrixAffine(
        const vk::AbstractCamera &cam,   //; 相机模型
        const Vector2d &px_ref,          //; 图像像素坐标
        const Vector3d &f_ref,           //; 归一化坐标
        const double depth_ref,          //; ref相机系下的向量的深度
        const SE3 &T_cur_ref,            //; ref帧和cur帧之间的位姿变换
        const int level_ref,       //; 实际输入0， // the corresponding pyrimid level of px_ref
        const int pyramid_level,   //; 实际输入0
        const int halfpatch_size,  //; patch的一半
        Matrix2d &A_cur_ref)   //; 输出结果，cur和ref之间的affine变换
    {
        // Compute affine warp matrix A_ref_cur
        const Vector3d xyz_ref(f_ref * depth_ref);  //; 中心点在相机系下的坐标
        Vector3d xyz_du_ref(
            cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
        Vector3d xyz_dv_ref(
            cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
        //   Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
        //   Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
        //; 下面的操作就是让du,dv偏移量的点在相机系下的深度和中心点的深度相等
        xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
        xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
        //; 把三个点转到cur帧的相机系下，然后再投影到cur帧的像素坐标系上
        const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
        const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
        const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
        // 最终的仿射矩阵
        A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
        A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
    }

    /**
     * @brief 输入ref帧图像，和上面的一个像素点，以及 cur和ref图像之间的affine变换，计算ref图像的一个patch
     *   affine到当前帧图像上之后的像素值
     */
    void LidarSelector::warpAffine(
        const Matrix2d &A_cur_ref,
        const cv::Mat &img_ref,
        const Vector2d &px_ref,
        const int level_ref,
        const int search_level,
        const int pyramid_level,   //; 一直都是0
        const int halfpatch_size,
        float *patch)   // patch是输出结果，也就是ref帧的像素affine到cur帧的图像之后的像素值
    {
        const int patch_size = halfpatch_size * 2;
        //; 计算 cur affine到 ref上的像素坐标，这是反向warp，这样可以对ref图像上的像素做插值
        const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
        if (isnan(A_ref_cur(0, 0)))
        {
            printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
            return;
        }
        //   Perform the warp on a larger patch.
        //   float* patch_ptr = patch;
        //   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref) / (1<<pyramid_level);
        //   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
        for (int y = 0; y < patch_size; ++y)
        {
            for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
            {
                // P[patch_size_total*level + x*patch_size+y]
                Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
                px_patch *= (1 << search_level);
                px_patch *= (1 << pyramid_level);
                //; A_ref_cur * px_patch 是计算 affine之后的像素偏移，
                const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>()); // 得到仿射变化后的patch坐标
                if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
                    patch[patch_size_total * pyramid_level + y * patch_size + x] = 0; // pyramid_level == 0
                // *patch_ptr = 0;
                else
                    //; 如果变换之后的像素坐标正常的话，对ref图像上的像素进行插值，得到affine之后的像素值
                    patch[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(
                        img_ref, px[0], px[1]);
                // *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
            }
        }
    }

    double LidarSelector::NCC(float *ref_patch, float *cur_patch, int patch_size)
    {
        double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
        double mean_ref = sum_ref / patch_size;

        double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
        double mean_curr = sum_cur / patch_size;

        double numerator = 0, demoniator1 = 0, demoniator2 = 0;
        for (int i = 0; i < patch_size; i++)
        {
            double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
            numerator += n;
            demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
            demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
        }
        return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
    }

    int LidarSelector::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
    {
        // Compute patch level in other image
        int search_level = 0;
        double D = A_cur_ref.determinant(); // 行列式（几何意义） D是仿射变化后四边形的面积

        //; D > 3.0 说明affine之后像素变大了，所以要到上一层的金字塔中进行匹配
        while (D > 3.0 && search_level < max_level)
        { // hr: make sure D <= 3 && search_level < 2
            search_level += 1;
            D *= 0.25;  //; 每一层面积缩小4倍，因为水平、垂直各缩小2倍
        }
        return search_level;
    }

    void LidarSelector::createPatchFromPatchWithBorder(float *patch_with_border, float *patch_ref)
    {
        float *ref_patch_ptr = patch_ref;
        for (int y = 1; y < patch_size + 1; ++y, ref_patch_ptr += patch_size)
        {
            float *ref_patch_border_ptr = patch_with_border + y * (patch_size + 2) + 1;
            for (int x = 0; x < patch_size; ++x)
                ref_patch_ptr[x] = ref_patch_border_ptr[x];
        }
    }

    /**
     * @brief 传入当前帧图像和上一帧的LIDAR点云，计算这些点云中选择为当前帧图像的某个网格中的点，并计算
     *   这个点附着的和当前帧观测角度最相近的patch，然后计算这个patch所在帧和当前帧之间的affine变换，
     *   然后计算patch所在帧在当前帧的图像下对应patch位置的像素值，最后计算patch和当前帧观测到的patch
     *   之间的像素误差
     * @param[in] img 
     * @param[in] pg 
     */
    void LidarSelector::addFromSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg)
    {
        if (feat_map.size() <= 0)
            return;
        /* 初始化深度地图 */
        double ts0 = omp_get_wtime();
        
        //; 先对上一帧的点云进行降采样
        pg_down->reserve(feat_map.size());
        downSizeFilter.setInputCloud(pg);
        downSizeFilter.filter(*pg_down);
        
        // Step 0.1: 把当前帧的子地图中的点全部清空
        reset_grid();
        memset(map_value, 0, sizeof(float) * length); // length = grid_n_width * grid_n_height
        cout << "1111111111111111111111: " << sizeof(float)*length << endl; // todo:???768

        // Step 0.2: 清空最终用到的子地图的值，这里面只存储了观测的patch等有用信息，而没有存储中间信息
        sub_sparse_map->reset();    //; reset是自定义函数，内部是把所有的成员变量都清空 
        deque<PointPtr>().swap(sub_map_cur_frame_);

        float voxel_size = 0.5;

        unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);  //; 首先清空当前帧包含的体素子地图
        unordered_map<int, Warp *>().swap(Warp_map);

        // cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
        // float* it = (float*)depth_img.data;
        
        // std::cout << "h*w = " << height * width << std::endl;
        //?bug: 这个地方如果按照下面C语言的写法有的数据集会报内存错误，这里改成std::vector就不会
        // float it[height * width] = {0.0};   //; 存储的图像中每个点的深度，这是后面对网格中的地图点检查深度连续性使用的
        std::vector<float> it(height*width, 0);

        double t_insert, t_depth, t_position;
        t_insert = t_depth = t_position = 0;

        int loc_xyz[3];

        // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);

        /*  */
        double ts1 = omp_get_wtime();
        
        // Step 1: 计算上一帧的 LiDAR 点云投影到当前帧图像下，给图像的点赋值深度，这是为了后面检查特征点深度连续性使用的
        for (int i = 0; i < pg_down->size(); i++)
        {
            // Transform Point to world coordinate
            V3D pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z); // 世界坐标系的点云坐标

            // Determine the key of hash table
            for (int j = 0; j < 3; j++)
            {
                loc_xyz[j] = floor(pt_w[j] / voxel_size); // voxel_size:0.5 floor:向下取整函数,取不超过x的最大整数
            }
            //; 当前LiDAR点的体素坐标
            VOXEL_KEY position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

            auto iter = sub_feat_map.find(position);
            if (iter == sub_feat_map.end())  // 没找到position
            {                         
                //; 感觉这里的表示的是 position 这个位置的体素在局部地图中，但是实际上从后面来看这里直接用set就可以，没必要用map
                sub_feat_map[position] = 1.0; // 未查找到，返回一个指向sub_feat_map.end()的指针，则需要初始化
            }

            //; 相机坐标系下的点
            V3D pt_c(new_frame_->w2f(pt_w)); // 世界坐标转换为相机frame

            V2D px;
            if (pt_c[2] > 0) // z > 0，也就是在相机的前面
            {
                //TODO: 这里直接投影了，没有计算畸变系数，是否会有影响？
                px[0] = fx * pt_c[0] / pt_c[2] + cx;
                px[1] = fy * pt_c[1] / pt_c[2] + cy;

                if (new_frame_->cam_->isInFrame(px.cast<int>(), (patch_size_half + 1) * 8))
                {
                    float depth = pt_c[2];
                    int col = int(px[0]);
                    int row = int(px[1]);
                    //; 给图像中这个像素点的深度赋值为LiDAR点的值，注意这里用的是上一帧LiDAR点云，因此应该不会有重复的点
                    it[width * row + col] = depth; // TODO：idx是如何定义的
                }
            }
        }

        //    imshow("depth_img", depth_img);
        //        printf("A1: %.6lf \n", omp_get_wtime() - ts1);
        //    printf("A11. calculate pt position: %.6lf \n", t_position);
        //    printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
        //    printf("A13. generate depth map: %.6lf \n", t_depth);
        // 投影
        // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

        /* B. feat_map.find */
        double t1 = omp_get_wtime();
        
        // Step 2: 遍历上面找到的所有体素，把体素中的所有地图点都拿出来投影到图像上；然后划分网格，保留网格中深度最近的那个点
        for (auto &iter : sub_feat_map)
        {
            VOXEL_KEY position = iter.first;  //; 这个体素的哈希值
            double t4 = omp_get_wtime();
            auto corre_voxel = feat_map.find(position);  //; 哈希值对应的体素
            double t5 = omp_get_wtime();

            //; 如果这个体素存在于地图中，则把体素中的点全部投影到当前帧图像上，寻找可以使用的地图点
            if (corre_voxel != feat_map.end())  
            {
                //; 这个体素中的所有点
                std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
                int voxel_num = voxel_points.size();
                for (int i = 0; i < voxel_num; i++)
                {
                    PointPtr pt = voxel_points[i];
                    if (pt == nullptr)
                        continue;
                    //; 把这个点转到相机系下 
                    V3D pt_cam(new_frame_->w2f(pt->pos_));
                    if (pt_cam[2] < 0)
                        continue;
                    //; 把这个点转到像素坐标系下，注意这个函数里面是考虑了相机的畸变的，因此是准确的
                    V2D pc(new_frame_->w2c(pt->pos_));

                    // FeaturePtr ref_ftr;  // 这里其实没有使用这个变量，可以注释掉

                    // 20px is the patch size in the matcher
                    //; 如果像素点在相机的FoV内
                    if (new_frame_->cam_->isInFrame(pc.cast<int>(),
                                                    (patch_size_half + 1) * 8)) 
                    {
                        //; 对像素点划分网格，计算这个像素点属于哪个网格
                        int index = static_cast<int>(pc[0] / grid_size) * grid_n_height +
                                    static_cast<int>(pc[1] / grid_size); // TODO: HOW?
                        grid_num[index] = TYPE_MAP;
                        
                        //; 当前点和相机之间构成的观测向量，注意仍然是在world系下表示的
                        Vector3d obs_vec(new_frame_->pos() - pt->pos_); // new_frame_->pos() return该帧在世界坐标系下的位姿 向量做差（3D点误差）

                        float cur_dist = obs_vec.norm(); // 向量范数，即差值模值
                        //; value 是 shiTomasiScore，也就是角点的得分，得分越高，说明这个角点越明显
                        float cur_value = pt->value;     // TODO:value?

                        //; 这里就是论文中说的，为了防止遮挡，会选 40x40 的grid中像素最小的那个点
                        if (cur_dist <= map_dist[index])
                        {
                            map_dist[index] = cur_dist; // map_dist 初始值 10000
                            voxel_points_[index] = pt;  //; 最终这个网格里存储的LiDAR点
                        }

                        //! 疑问：value是存储的什么？
                        if (cur_value >= map_value[index])
                        {
                            map_value[index] = cur_value; // map_value 初始值 0
                        }
                    }
                }
            }
        }

        double t2 = omp_get_wtime();

        //        cout << "B. feat_map.find: " << t2 - t1 << endl;

        /* C. addSubSparseMap: */
        double t_2, t_3, t_4, t_5;
        t_2 = t_3 = t_4 = t_5 = 0;

        // Step 3: 遍历上面的所有网格，如果里面有3D点，那么进一步处理看是否要把这个3D点作为最后的观测
        for (int i = 0; i < length; i++)
        {
            //; 如果这个网格类型是 TYPE_MAP，说明在上一步中找到了一个LiDAR点投影到图像后落在这个网格里
            if (grid_num[i] == TYPE_MAP) //&& map_value[i]>10)
            {
                double t_1 = omp_get_wtime();

                PointPtr pt = voxel_points_[i];  //; 取出这个网格里存储的LIDAR点

                if (pt == nullptr)
                    continue;

                //; 再次把这个LIDAR点投影到 像素系 和 相机系
                V2D pc(new_frame_->w2c(pt->pos_));     // world frame to camera pixel coordinates（2d）
                V3D pt_cam(new_frame_->w2f(pt->pos_)); // world frame to camera frame

                // Step 3.1: 判断点深度连续性，即当前点其周围8个patch像素的深度差别不应该太大
                bool depth_continous = false;
                for (int u = -patch_size_half; u <= patch_size_half; u++)
                {
                    for (int v = -patch_size_half; v <= patch_size_half; v++)
                    {
                        if (u == 0 && v == 0)
                            continue; // patch中心

                        // int col = int(px[0]);int row = int(px[1]);it[width*row+col] = depth;
                        float depth = it[width * (v + int(pc[1])) + u + int(pc[0])]; 

                        if (depth == 0.)
                            continue;

                        double delta_dist = abs(pt_cam[2] - depth);

                        //; 当前点和它周围任何一个点的深度超过1.5m，则深度不连续，直接跳出
                        if (delta_dist > 1.5)  
                        {
                            depth_continous = true;
                            break;
                        }
                    }
                    if (depth_continous)
                        break;
                }
                //; 如果深度不连续，则跳过当前点，不使用它
                if (depth_continous)
                    continue;

                t_1 = omp_get_wtime();


                // Step 3.2: 寻找这个地图点的所有patch中，和当前的图像的观测角度最相近的那个patch
                FeaturePtr ref_ftr;  //; 这个就是当前帧图像匹配的地图中的 patch

                // get frame with same point of view AND same pyramid level
                //; 找到与当前图像的观察角度最接近的点作为参考，注意里面只用观测方向进行了判断
                if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc))
                    continue; // <= 60度

                //                t_3 += omp_get_wtime() - t_1;

                //; 这里 patch_size_total 是patch占用的所有像素，比如 8*8=64
                //! 疑问： *3是干什么的？
                //; 解答：因为有图像金字塔，缩放两次，加上原始图像，所一共有3个wrap
                float *patch_wrap = new float[patch_size_total * 3];  //; 地图中匹配的patch像素

                patch_wrap = ref_ftr->patch;  //! 疑问：怎么又换了指向的方向了？

                t_1 = omp_get_wtime();

                int search_level;
                Matrix2d A_cur_ref_zero;

                // Step 3.3: 计算这个patch所在的图像帧和当前帧的图像像素之间的affine变换
                //; 这个 ref_ftr->id_ 应该是这个patch所在的图像的id，因为对于一个图像来说和当前帧的affine变换应该是一样的
                auto iter_warp = Warp_map.find(ref_ftr->id_);
                if (iter_warp != Warp_map.end())  // find sucessfully
                { 
                    search_level = iter_warp->second->search_level;  //; 地图中这个 patch 对应图像和当前图像之间的 warp 的金字塔
                    A_cur_ref_zero = iter_warp->second->A_cur_ref;
                }
                else
                {
                    // 计算仿射矩阵 计算 patch 从参考帧投影到当前帧的仿射变换，因为要计算 patch 的相似度
                    getWarpMatrixAffine(*cam, ref_ftr->px, ref_ftr->f, (ref_ftr->pos() - pt->pos_).norm(),
                                        new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0, patch_size_half,
                                        A_cur_ref_zero);

                    //; 判断到哪个层里面寻找像素对应关系
                    search_level = getBestSearchLevel(A_cur_ref_zero, 2); // 找到尺度相近的层
                    Warp *ot = new Warp(search_level, A_cur_ref_zero);
                    Warp_map[ref_ftr->id_] = ot; // 更新warp_map
                }

                //                t_4 += omp_get_wtime() - t_1;

                t_1 = omp_get_wtime();

                // Step 3.4: 利用affien变换，计算ref帧的patch变换到当前帧的图像之后的像素值
                for (int pyramid_level = 0; pyramid_level <= 0; pyramid_level++) // pyramid_level == 0
                { 
                    //! 注意：这里算的是反向的warp，和深度估计里面差不多
                    // 只对第0层实施仿射变换，可以得到亚像素级别的精度
                    warpAffine(A_cur_ref_zero, ref_ftr->img, ref_ftr->px, ref_ftr->level, 
                               search_level, pyramid_level, patch_size_half, patch_wrap); 
                }

                // Step 3.5: 对当前帧的图像取patch，得到patch中的像素值
                getpatch(img, pc, patch_cache, 0);  //; 最后0表示原始图像，即没有使用图像金字塔

                if (ncc_en)  // false
                { 
                    double ncc = NCC(patch_wrap, patch_cache, patch_size_total);
                    if (ncc < ncc_thre)
                        continue;
                }

                // Step 3.6: 计算ref帧的patch和cur帧的patch之间的像素误差
                float error = 0.0;
                for (int ind = 0; ind < patch_size_total; ind++)
                {
                     // (ref-current)^2
                    error += (patch_wrap[ind] - patch_cache[ind]) * (patch_wrap[ind] - patch_cache[ind]);
                }

                // Step 3.7: 如果这个误差过大，说明可能是误匹配，那么就不要这个patch对的观测了
                if (error > outlier_threshold * patch_size_total)
                    continue; // TODO：阈值：外点去除？

                // Step 3.8: 收尾工作，当前点可以用，把所有的信息收集起来
                sub_map_cur_frame_.push_back(pt);   //; 把这个LiDAR地图点存储起来，实际上不是优化过程中用的

                //; 把当前这个点加到视觉稀疏子地图中，视觉稀疏子地图才是后面真正要用的
                sub_sparse_map->align_errors.push_back(error);
                sub_sparse_map->propa_errors.push_back(error);
                sub_sparse_map->search_levels.push_back(search_level);  //; 这个地图点的patch要和当前帧的patch的哪个金字塔层匹配
                sub_sparse_map->errors.push_back(error);     //; 历史上的所有误差？ 
                sub_sparse_map->index.push_back(i);          //; 这个地图点落在图像中的哪个网格里
                sub_sparse_map->voxel_points.push_back(pt);  //; 把这个LiDAR点存到子地图中
                sub_sparse_map->patch.push_back(patch_wrap); //; 把这个LiDAR点对应的patch进行affine到当前帧之后的像素值存储下来
                // sub_sparse_map->px_cur.push_back(pc);
                // sub_sparse_map->propa_px_cur.push_back(pc);
                //                t_5 += omp_get_wtime() - t_1;
            }
        }
        double t3 = omp_get_wtime();
        //        cout << "C. addSubSparseMap: " << t3 - t2 << endl;
        //        cout << "depthcontinuous: C1 " << t_2 << " C2 " << t_3 << " C3 " << t_4 << " C4 " << t_5 << endl;
        //        cout << "[ addFromSparseMap ] " << sub_sparse_map->index.size() << endl;
        // printf("[ VIO ]: choose %d points from sub_sparse_map.\n", int(sub_sparse_map->index.size()));
    }

    bool LidarSelector::align2D(
        const cv::Mat &cur_img,
        float *ref_patch_with_border,
        float *ref_patch,
        const int n_iter,
        Vector2d &cur_px_estimate,
        int index)
    {
#ifdef __ARM_NEON__
        if (!no_simd)
            return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
#endif

        const int halfpatch_size_ = 4;
        const int patch_size_ = 8;
        const int patch_area_ = 64;
        bool converged = false;

        // compute derivative of template and prepare inverse compositional
        float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
        float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
        Matrix3f H;
        H.setZero();

        // compute gradient and hessian
        const int ref_step = patch_size_ + 2;
        float *it_dx = ref_patch_dx;
        float *it_dy = ref_patch_dy;
        for (int y = 0; y < patch_size_; ++y)
        {
            float *it = ref_patch_with_border + (y + 1) * ref_step + 1;
            for (int x = 0; x < patch_size_; ++x, ++it, ++it_dx, ++it_dy)
            {
                Vector3f J;
                J[0] = 0.5 * (it[1] - it[-1]);
                J[1] = 0.5 * (it[ref_step] - it[-ref_step]);
                J[2] = 1;
                *it_dx = J[0];
                *it_dy = J[1];
                H += J * J.transpose();
            }
        }
        Matrix3f Hinv = H.inverse();
        float mean_diff = 0;

        // Compute pixel location in new image:
        float u = cur_px_estimate.x();
        float v = cur_px_estimate.y();

        // termination condition
        const float min_update_squared = 0.03 * 0.03; //0.03*0.03
        const int cur_step = cur_img.step.p[0];
        float chi2 = 0;
        chi2 = sub_sparse_map->propa_errors[index];
        Vector3f update;
        update.setZero();
        for (int iter = 0; iter < n_iter; ++iter)
        {
            int u_r = floor(u);
            int v_r = floor(v);
            if (u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols - halfpatch_size_ ||
                v_r >= cur_img.rows - halfpatch_size_)
                break;

            if (isnan(u) ||
                isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
                return false;

            // compute interpolation weights
            float subpix_x = u - u_r;
            float subpix_y = v - v_r;
            float wTL = (1.0 - subpix_x) * (1.0 - subpix_y);
            float wTR = subpix_x * (1.0 - subpix_y);
            float wBL = (1.0 - subpix_x) * subpix_y;
            float wBR = subpix_x * subpix_y;

            // loop through search_patch, interpolate
            float *it_ref = ref_patch;
            float *it_ref_dx = ref_patch_dx;
            float *it_ref_dy = ref_patch_dy;
            float new_chi2 = 0.0;
            Vector3f Jres;
            Jres.setZero();
            for (int y = 0; y < patch_size_; ++y)
            {
                uint8_t *it = (uint8_t *)cur_img.data + (v_r + y - halfpatch_size_) * cur_step + u_r - halfpatch_size_;
                for (int x = 0; x < patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
                {
                    float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
                    float res = search_pixel - *it_ref + mean_diff;
                    Jres[0] -= res * (*it_ref_dx);
                    Jres[1] -= res * (*it_ref_dy);
                    Jres[2] -= res;
                    new_chi2 += res * res;
                }
            }

            if (iter > 0 && new_chi2 > chi2)
            {
                //   cout << "error increased." << endl;
                u -= update[0];
                v -= update[1];
                break;
            }
            chi2 = new_chi2;

            sub_sparse_map->align_errors[index] = new_chi2;

            update = Hinv * Jres;
            u += update[0];
            v += update[1];
            mean_diff += update[2];

#if SUBPIX_VERBOSE
            cout << "Iter " << iter << ":"
                 << "\t u=" << u << ", v=" << v
                 << "\t update = " << update[0] << ", " << update[1]
            //         << "\t new chi2 = " << new_chi2 << endl;
#endif

                if (update[0] * update[0] + update[1] * update[1] < min_update_squared)
            {
#if SUBPIX_VERBOSE
                cout << "converged." << endl;
#endif
                converged = true;
                break;
            }
        }

        cur_px_estimate << u, v;
        return converged;
    }

    void LidarSelector::FeatureAlignment(cv::Mat img)
    {
        int total_points = sub_sparse_map->index.size();
        if (total_points == 0)
            return;
        memset(align_flag, 0, length);
        int FeatureAlignmentNum = 0;

        for (int i = 0; i < total_points; i++)
        {
            bool res;
            int search_level = sub_sparse_map->search_levels[i];
            Vector2d px_scaled(sub_sparse_map->px_cur[i] / (1 << search_level));
            res = align2D(new_frame_->img_pyr_[search_level], sub_sparse_map->patch_with_border[i],
                          sub_sparse_map->patch[i],
                          20, px_scaled, i);
            sub_sparse_map->px_cur[i] = px_scaled * (1 << search_level);
            if (res)
            {
                align_flag[i] = 1;
                FeatureAlignmentNum++;
            }
        }
    }

    /**
     * @brief 利用patch之间的光度误差优化状态的主函数！
     * 
     * @param[in] img 
     * @param[in] total_residual 
     * @param[in] level 从第几层开始优化，因为光度误差非凸性太强了，要使用金字塔逐层的优化
     * @return float 
     */
    float LidarSelector::UpdateState(cv::Mat img, float total_residual, int level)
    {
        int total_points = sub_sparse_map->index.size();
        if (total_points == 0)
            return 0.;
        StatesGroup old_state = (*state);
        V2D pc;
        MD(1, 2) Jimg;
        MD(2, 3) Jdpi;
        MD(1, 3) Jdphi, Jdp, JdR, Jdt;
        VectorXd z;
        // VectorXd R;
        bool EKF_end = false;
        /* Compute J */
        float error = 0.0, last_error = total_residual, patch_error = 0.0, last_patch_error = 0.0, propa_error = 0.0;
        // MatrixXd H;
        bool z_init = true;
        const int H_DIM = total_points * patch_size_total;  //; 残差的维度，点的总数 * patch大小

        // K.resize(H_DIM, H_DIM);
        z.resize(H_DIM);  
        z.setZero();
        // R.resize(H_DIM);
        // R.setZero();

        // H.resize(H_DIM, DIM_STATE);
        // H.setZero();
        H_sub.resize(H_DIM, 6);  //; 这里写成H_sub是因为光度残差之和位姿有关，所以6维就够了
        H_sub.setZero();

        for (int iteration = 0; iteration < NUM_MAX_ITERATIONS; iteration++)
        { // NUM_MAX_ITERATIONS:4(default);10(yaml)
            double t1 = omp_get_wtime();
            double count_outlier = 0;

            error = 0.0;
            propa_error = 0.0;
            n_meas_ = 0;
            M3D Rwi(state->rot_end);
            V3D Pwi(state->pos_end);
            //; 根据IMU预测的位姿，计算当前帧图像在world系下的位置
            Rcw = Rci * Rwi.transpose();
            Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
            Jdp_dt = Rci * Rwi.transpose();

            M3D p_hat;
            int i;

            for (i = 0; i < sub_sparse_map->index.size(); i++)
            {
                patch_error = 0.0;
                //; search_level是patch匹配要求的level，而level是当前优化从哪个level开始
                int search_level = sub_sparse_map->search_levels[i];
                int pyramid_level = level + search_level;
                const int scale = (1 << pyramid_level); // 2^pyramid_level

                PointPtr pt = sub_sparse_map->voxel_points[i];  //; 3D 地图点

                if (pt == nullptr)
                    continue;
                
                //; 把地图点从 world 系投到当前帧 相机系
                V3D pf = Rcw * pt->pos_ + Pcw; // pt: world frame; pf: camera frame
                pc = cam->world2cam(pf);  //; pc是patch的中心点的像素坐标
                // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                {
                    //; 十四讲 P220, (8.16), du/dq
                    dpi(pf, Jdpi);               // use pf(x,y,z) to return MD(2, 3) Jdpi
                    //; 十四讲 P220, (8.17), dq/dT
                    p_hat << SKEW_SYM_MATRX(pf); // 0.0, -pf[2], pf[1], pf[2], 0.0, -pf[0], -pf[1], pf[0], 0.0
                }
                const float u_ref = pc[0];
                const float v_ref = pc[1];
                const int u_ref_i = floorf(pc[0] / scale) * scale; 
                const int v_ref_i = floorf(pc[1] / scale) * scale;
                const float subpix_u_ref = (u_ref - u_ref_i) / scale;
                const float subpix_v_ref = (v_ref - v_ref_i) / scale;
                const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
                const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
                const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
                const float w_ref_br = subpix_u_ref * subpix_v_ref;

                float *P = sub_sparse_map->patch[i];  //; 取出地图观测的patch
                //; x是遍历当前patch的纵坐标，y是遍历当前patch的横坐标
                for (int x = 0; x < patch_size; x++) 
                {
                    //; 取当前帧的图像的像素在对应的金字塔层上的像素坐标值
                    uint8_t *img_ptr =
                        (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width 
                        + u_ref_i - patch_size_half * scale; // TODO:?
                    for (int y = 0; y < patch_size; ++y, img_ptr += scale)
                    {
                        // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                        //{
                        //; 这里就是对当前帧图像上的点进行线性插值，然后计算像素梯度
                        float du = 0.5f * ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] +
                                            w_ref_bl * img_ptr[scale * width + scale] +
                                            w_ref_br * img_ptr[scale * width + scale * 2]) -
                                           (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] +
                                            w_ref_bl * img_ptr[scale * width - scale] +
                                            w_ref_br * img_ptr[scale * width]));
                        float dv = 0.5f *
                                   ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] +
                                     w_ref_bl * img_ptr[width * scale * 2] +
                                     w_ref_br * img_ptr[width * scale * 2 + scale]) -
                                    (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] +
                                     w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));
                        Jimg << du, dv;  //; 像素梯度雅克比
                        Jimg = Jimg * (1.0 / scale);  //; 这里除以尺度是因为这个像素坐标是金字塔缩小之后的，所以梯度也会缩小
                        //; 这个是de/dR, 是对旋转的李代数导数
                        Jdphi = Jimg * Jdpi * p_hat;  //; 十四讲P200，（8.19）
                        //; 这个是de/dt，是对平移的李代数导数
                        Jdp = -Jimg * Jdpi;

                        //; 上面都是对相机系的位姿的雅克比，这里还要转成对IMU系的位姿的雅克比
                        //! 疑问：怎么推导的？
                        JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
                        Jdt = Jdp * Jdp_dt;

                        //; 这里就是计算当前帧图像的像素和patch像素之间的残差。注意这里和雅克比的定义恰好差符号，因为后面
                        //; 正规方程中使用的z就是差符号的，也就是正常是Hx = -b，而作者用的是 Hx = b
                        double res =
                            w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + 
                            w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale] -
                            P[patch_size_total * level + x * patch_size + y]; // TODO:公式推导

                        //; 存储残差
                        z(i * patch_size_total + x * patch_size + y) = res;  
                        // float weight = 1.0;
                        // if(iteration > 0)
                        //     weight = weight_function_->value(res/weight_scale_);
                        // R(i*patch_size_total+x*patch_size+y) = weight;
                        patch_error += res * res;
                        n_meas_++;
                        // H.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR*weight, Jdt*weight;
                        // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                        //; 存储雅克比
                        H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
                    }
                }

                sub_sparse_map->errors[i] = patch_error;
                error += patch_error;
            }

            //            computeH += omp_get_wtime() - t1;

            error = error / n_meas_;

            double t3 = omp_get_wtime();

            if (error <= last_error)
            {
                old_state = (*state);
                last_error = error;

                // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov;
                // auto vec = (*state_propagat) - (*state);
                // G = K*H;
                // (*state) += (-K*z + vec - G*vec);

                auto &&H_sub_T = H_sub.transpose();  //; 6*n
                H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
                MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H +
                                                  (state->cov / img_point_cov).inverse())
                                                     .inverse(); // TODO：视觉协方差
                auto &&HTz = H_sub_T * z;  //; 6*n x n*1 = 6*1
                // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
                //; state_propagat 就是IMU预测的状态，而state是当前状态，所以vec就是状态的误差，这个就是IEKF的公式
                auto vec = (*state_propagat) - (*state);
                G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
                //! 疑问：感觉这里多了一项vec? 应该是没有vec的吧？
                auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec -
                                G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
                (*state) += solution;
                auto &&rot_add = solution.block<3, 1>(0, 0);
                auto &&t_add = solution.block<3, 1>(3, 0);

                if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
                { // TODO:EKF结束判断阈值(视觉约束阈值)
                    EKF_end = true;
                }
            }
            else
            {
                (*state) = old_state;
                EKF_end = true;
            }

            //            ekf_time += omp_get_wtime() - t3;

            if (iteration == NUM_MAX_ITERATIONS || EKF_end)
            {
                break;
            }
        }
        return last_error;
    }

    void LidarSelector::updateFrameState(StatesGroup state)
    {
        M3D Rwi(state.rot_end);
        V3D Pwi(state.pos_end);
        Rcw = Rci * Rwi.transpose();
        Pcw = -Rci * Rwi.transpose() * Pwi + Pci; // hr: world to camera
        new_frame_->T_f_w_ = SE3(Rcw, Pcw);
    }

    /**
     * @brief 优化结束后，添加当前帧对视觉地图点的新的观测
     * 
     */
    void LidarSelector::addObservation(cv::Mat img)
    {
        int total_points = sub_sparse_map->index.size();
        if (total_points == 0)
            return;

        for (int i = 0; i < total_points; i++)
        {
            PointPtr pt = sub_sparse_map->voxel_points[i];   //; 观测到的地图点
            if (pt == nullptr)
                continue;
            V2D pc(new_frame_->w2c(pt->pos_));  //; 地图点投影到当前帧
            SE3 pose_cur = new_frame_->T_f_w_;
            bool add_flag = false;
            // if (sub_sparse_map->errors[i]<= 100*patch_size_total && sub_sparse_map->errors[i]>0) //&& align_flag[i]==1)
            {
                float *patch_temp = new float[patch_size_total * 3];
                getpatch(img, pc, patch_temp, 0);
                getpatch(img, pc, patch_temp, 1);
                getpatch(img, pc, patch_temp, 2);

                //TODO: condition: distance and view_angle
                // Step 1: time
                FeaturePtr last_feature = pt->obs_.back(); // 最新的feature
                // if(new_frame_->id_ >= last_feature->id_ + 20) add_flag = true;

                // Step 2: delta_pose
                //; 计算当前帧和地图点最新观测帧之间的位姿变换，如果超过阈值，则加入当前帧对地图点的观测
                SE3 pose_ref = last_feature->T_f_w_;
                SE3 delta_pose = pose_ref * pose_cur.inverse();
                double delta_p = delta_pose.translation().norm();
                double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));
                if (delta_p > 0.5 || delta_theta > 10)
                    add_flag = true;

                // Step 3: pixel distance
                //; 判断当前帧和地图点最近观测帧之间的像素差，如果超过阈值，则加入当前帧对地图点的观测
                Vector2d last_px = last_feature->px;
                double pixel_dist = (pc - last_px).norm();
                if (pixel_dist > 40)
                    add_flag = true;

                // Maintain the size of 3D Point observation features.
                //; 如果地图点的观测个数过多，那么删除之前的观测，保留最近比较新的观测
                if (pt->obs_.size() >= 20)
                {
                    FeaturePtr ref_ftr;
                    pt->getFurthestViewObs(new_frame_->pos(), ref_ftr);
                    pt->deleteFeatureRef(ref_ftr);
                    // ROS_WARN("ref_ftr->id_ is %d", ref_ftr->id_);
                }
                //; 如果添加新的观测，则new一个Feature，并把它添加到地图点中
                if (add_flag)
                {
                    pt->value = vk::shiTomasiScore(img, pc[0], pc[1]);
                    Vector3d f = cam->cam2world(pc);
                    FeaturePtr ftr_new(new Feature(patch_temp, pc, f, new_frame_->T_f_w_, pt->value,
                                                   sub_sparse_map->search_levels[i]));
                    ftr_new->img = new_frame_->img_pyr_[0];
                    ftr_new->id_ = new_frame_->id_;
                    // ftr_new->ImgPyr.resize(5);
                    // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
                    pt->addFrameRef(ftr_new);
                }
            }
        }
    }


    void LidarSelector::ComputeJ(cv::Mat img)
    {
        int total_points = sub_sparse_map->index.size();  //; 子地图找到的所有的匹配的patch个数
        if (total_points == 0)
            return;
        float error = 1e10;
        float now_error = error;

        // Step :视觉优化的主函数！三次循环，coarse-to-fine的三次优化
        for (int level = 2; level >= 0; level--)
        { // hr: a coarse-to-fine manner 2->0:粗糙->精细
            now_error = UpdateState(img, error, level);
        }

        //; 如果误差降低，那么才把最终的协方差矩阵进行更新
        if (now_error < error)
        {
            state->cov -= G * state->cov; // 更新协方差
        }
        updateFrameState(*state); // get: new_frame_->T_f_w_
    }

    void LidarSelector::display_keypatch(double time)
    {
        int total_points = sub_sparse_map->index.size();
        if (total_points == 0)
            return;
        for (int i = 0; i < total_points; i++)
        {
            PointPtr pt = sub_sparse_map->voxel_points[i];
            V2D pc(new_frame_->w2c(pt->pos_));
            cv::Point2f pf;
            pf = cv::Point2f(pc[0], pc[1]);   //; 把选择的3D地图点投影到当前帧的图像上
            //! 疑问：这个8000的值是怎么计算的呢？
            if (sub_sparse_map->errors[i] < 8000)   // 5.5
                cv::circle(img_cp, pf, 4, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
            else
                cv::circle(img_cp, pf, 4, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
        }
        //; 这个hz其实就反映了VIO这个部分能够处理的最高频率，因为这个频率是纯算VIO的时间得到的频率
        std::string text = std::to_string(int(1 / time)) + " HZ";
        cv::Point2f origin;
        origin.x = 20;
        origin.y = 20;
        // 待绘制的图像；待绘制的文字；文本框左下角；字体；字体大小；线条颜色；.....
        cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
    }

    V3F LidarSelector::getpixel(cv::Mat img, V2D pc)
    {
        const float u_ref = pc[0];
        const float v_ref = pc[1];
        const int u_ref_i = floorf(pc[0]);
        const int v_ref_i = floorf(pc[1]);
        const float subpix_u_ref = (u_ref - u_ref_i);
        const float subpix_v_ref = (v_ref - v_ref_i);
        const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
        const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
        const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
        const float w_ref_br = subpix_u_ref * subpix_v_ref;
        uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
        float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] +
                  w_ref_br * img_ptr[width * 3 + 0 + 3];
        float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] +
                  w_ref_br * img_ptr[width * 3 + 1 + 3];
        float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] +
                  w_ref_br * img_ptr[width * 3 + 2 + 3];
        V3F pixel(B, G, R);
        return pixel;
    }

    /**
     * @brief 传入当前帧的图像，和上一帧的LiDAR在世界坐标系下的点云(主要用它所占用的体素来选择当前帧
     *   的FoV内的子地图)
     * 
     * @param[in] img  当前帧的图像
     * @param[in] pg   上一帧LiDAR扫描到的点在world系下的表示
     */
    void LidarSelector::detect(cv::Mat img, PointCloudXYZI::Ptr pg)
    {
        if (width != img.cols || height != img.rows)
        {
            // std::cout << "Resize the img scale !!!" << std::endl;
            //! 疑问：这里scale为什么直接给了0.5?
            double scale = 0.5;
            cv::resize(img, img, cv::Size(img.cols * scale, img.rows * scale), 0, 0, CV_INTER_LINEAR);
        }
        //; 这个是用于
        img_rgb = img.clone(); // 完全拷贝
        img_cp = img.clone();
        //! 疑问：为什么要进行颜色空间的转换？
        //; 解答：应该是以为要使用光度误差，所以要从彩色图转成灰度图，所以这里色彩空间是BGR2GRAY
        cv::cvtColor(img, img, CV_BGR2GRAY); // 图像从一个颜色空间转换到另一个颜色空间的转换 opencv default:BGR

        // Step 1: 使用相机模型和当前帧图像，构造一个图像帧，这个是在地图中维护的数据结构
        //; 注意这里有clone
        new_frame_.reset(new Frame(cam, img.clone()));
        
        //; 利用IMU积分预测得到的当前IMU在world系下的位姿，然后得到当前时刻下 world系 在 相机系 下的位姿
        updateFrameState(*state); // get transformation of world to camera ：T_f_w

        //; 如果当前帧是第一帧，并且点云足够，则设置当前帧为关键帧: 关键帧会寻找特征点
        //! 疑问：目前来看，只有第一帧图像会提取特征点？为什么要这样呢？按理来说第一帧不提特征点，
        //; 然后优化之后自然也会把第一帧对地图点的观测patch加到地图里啊？
        if (stage_ == STAGE_FIRST_FRAME && pg->size() > 10)
        {
            new_frame_->setKeyframe(); // hr: find feature points(5 points method)
            stage_ = STAGE_DEFAULT_FRAME;
        }

        double t1 = omp_get_wtime();

        // Step 2: 计算这个图像帧观测到的地图点的patch，结果会存到成员变量的 sub_sparse_map 里面。
        // 另外这里面就计算了地图点的patch和当前帧的patch之间的光度误差了，并且存到了 sub_sparse_map 里面
        addFromSparseMap(img, pg);

        double t3 = omp_get_wtime();

        // ADD required 3D points
        // Step 3: 这里属于论文最后一步，也就是优化之后添加新的地图点到视觉地图中。
        //         但是由于新的地图点和优化无关，所以这里提前添加也可以
        addSparseMap(img, pg);

        double t4 = omp_get_wtime();

        computeH = ekf_time = 0.0;

        // Step 4: 使用视觉直接对齐，对状态进行优化，内容并不难，可以参照十四讲直接法的章节
        ComputeJ(img); // EKF迭代误差更新状态

        double t5 = omp_get_wtime();
        
        // Step 5: 优化之后，对视觉地图点添加当前帧图像的新的patch观测
        // 根据distance and view_angle,添加新特征
        addObservation(img);

        double t2 = omp_get_wtime();

        //        cout << "addFromSparseMap time:" << t3 - t1 << endl;
        //        cout << "addSparseMap time: " << t4 - t3 << endl;
        //        cout << "ComputeJ time: " << t5 - t4 << " comp H: " << computeH << " ekf: " << ekf_time << endl;
        //        cout << "addObservation time: " << t2 - t5 << endl;

        frame_cont++;
        ave_total = ave_total * (frame_cont - 1) / frame_cont + (t2 - t1) / frame_cont;

        //        cout << "total time: " << t2 - t1 << " ave: " << ave_total << endl;
        // printf("[ VIO ]: time: addFromSparseMap: %0.6f addSparseMap: %0.6f ComputeJ: %0.6f addObservation: %0.6f total time: %0.6f ave_total: %0.6f.\n",
        //        t3 - t1, t4 - t3, t5 - t4, t2 - t5, t2 - t1);

        display_keypatch(t2 - t1); // 绘制关键patch
    }

} // namespace lidar_selection