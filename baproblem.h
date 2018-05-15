#pragma once

#include "parametersse3.h"
#include "utility.h"

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template <int PoseBlockSize>
class BAProblem
{
public:
    BAProblem(int pose_num_, int point_num_, double pix_noise_, bool useOrdering = true);

    void solve(ceres::Solver::Options &opt, ceres::Solver::Summary* sum);
    void validate();

    ceres::Problem problem;
    ceres::ParameterBlockOrdering* ordering;

protected:
    PosePointParametersBlock<PoseBlockSize> states;
    PosePointParametersBlock<PoseBlockSize> true_states;
    std::vector<bool> vbInOpt;

};


template<int PoseBlockSize>
BAProblem<PoseBlockSize>::BAProblem(int pose_num_, int point_num_, double pix_noise_, bool useOrdering)
{
    if(useOrdering)
        ordering = new ceres::ParameterBlockOrdering;
    else
        ordering = 0;

    int pose_num = pose_num_;
    int point_num = point_num_;
    double PIXEL_NOISE = pix_noise_;

    states.create(pose_num, point_num);
    true_states.create(pose_num, point_num);
    vbInOpt.resize(point_num);

    for (int i = 0; i < point_num; ++i)
    {
        Eigen::Map<Vector3d> true_pt(true_states.point(i));
        true_pt = Vector3d((Utility::uniform() - 0.5) * 3,
                           Utility::uniform() - 0.5,
                           Utility::uniform() + 3);
    }

    double focal_length = 1000.;
    double cx = 320.;
    double cy = 240.;
    CameraParameters cam(focal_length, cx, cy);

    for (int i = 0; i < pose_num; ++i)
    {
        Vector3d trans(i * 0.04 - 1., 0.0, 0.0);

        Eigen::Quaterniond q;
        q.setIdentity();
        true_states.setPose(i, q, trans);
        states.setPose(i, q, trans);

        problem.AddParameterBlock(states.pose(i), PoseBlockSize, new PoseSE3Parameterization<PoseBlockSize>());

        if(i < 2)
        {
            problem.SetParameterBlockConstant(states.pose(i));
        }
    }

    for (int i = 0; i < point_num; ++i)
    {
        vbInOpt[i] = false;
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        Eigen::Map<Vector3d> noise_point_i(states.point(i));
        noise_point_i = true_point_i + Vector3d(Utility::gaussian(1),
                                                Utility::gaussian(1),
                                                Utility::gaussian(1));

        Vector2d z;
        SE3 true_pose_se3;

        int num_obs = 0;
        for (int j = 0; j < pose_num; ++j)
        {
            true_states.getPose(j, true_pose_se3.rotation(), true_pose_se3.translation());
            Vector3d point_cam = true_pose_se3.map(true_point_i);
            z = cam.cam_map(point_cam);
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
            {
                ++num_obs;
            }
        }
        if (num_obs >= 2)
        {
            vbInOpt[i] = true;
            problem.AddParameterBlock(states.point(i), 3);
            if(useOrdering)
                ordering->AddElementToGroup(states.point(i), 0);

            for (int j = 0; j < pose_num; ++j)
            {
                true_states.getPose(j, true_pose_se3.rotation(), true_pose_se3.translation());
                Vector3d point_cam = true_pose_se3.map(true_point_i);
                z = cam.cam_map(point_cam);

                if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
                {
                    z += Vector2d(Utility::gaussian(PIXEL_NOISE),
                                  Utility::gaussian(PIXEL_NOISE));

                    ceres::CostFunction* costFunc = new ReprojectionErrorSE3XYZ<PoseBlockSize>(focal_length, cx, cy, z[0], z[1]);
                    problem.AddResidualBlock(costFunc, NULL, states.pose(j), states.point(i));
                }
            }

        }
    }

    if(useOrdering)
        for (int i = 0; i < pose_num; ++i)
        {
            ordering->AddElementToGroup(states.pose(i), 1);
        }

}


template<int PoseBlockSize>
void BAProblem<PoseBlockSize>::solve(ceres::Solver::Options& opt, ceres::Solver::Summary *sum)
{
    if(ordering != NULL)
        opt.linear_solver_ordering.reset(ordering);
    ceres::Solve(opt, &problem, sum);
}

template<int PoseBlockSize>
void BAProblem<PoseBlockSize>::validate()
{
    double cost = 0.0;
    for(int i = 0; i < vbInOpt.size(); ++i){
       if(vbInOpt[i]){
           Eigen::Map<Vector3d> est_pt(states.point(i));
           Eigen::Map<Vector3d> true_pt(true_states.point(i));
           cost += (est_pt-true_pt).norm();
       }
    }
    std::cout << "Total cost: " << cost << std::endl;
}

