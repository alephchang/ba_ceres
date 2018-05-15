#pragma once
#include "inverse_depth_factor.h"
#include "parametersse3.h"
#include "utility.h"

template<int PoseBlockSize>
class BAProblemIDP
{
public:
    BAProblemIDP(int pose_num_, int point_num_, double pix_noise_, bool useOrdering = true);

    void solve(ceres::Solver::Options &opt, ceres::Solver::Summary* sum);

    void validate();

    ceres::Problem problem;
    ceres::ParameterBlockOrdering* ordering;

protected:
    PosePointParametersBlock<PoseBlockSize> true_states;

    PosePointIDPParametersBlock<PoseBlockSize> states; // states.point(i) is inv_depth
    std::vector<std::pair<int, Vector2d> > ref_pose_obs;//<ref_frame_id, pix in ref frame>
};

template<int PoseBlockSize>
BAProblemIDP<PoseBlockSize>::BAProblemIDP(int pose_num_, int point_num_, double pix_noise_, bool useOrdering)
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

        if(i < 1)
        {
            problem.SetParameterBlockConstant(states.pose(i));
        }
    }
    ref_pose_obs.resize(point_num);
    for (int i = 0; i < point_num; ++i)
    {
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        Eigen::Vector3d noise_point_i;
        noise_point_i = true_point_i + Vector3d(Utility::gaussian(1),
                                                Utility::gaussian(1),
                                                Utility::gaussian(1));

        Vector2d z;
        SE3 true_pose_se3;
        ref_pose_obs[i].first = -1;
        std::vector<std::pair<int, Vector2d> > pose_obs;
        double inv_depth;
        bool ref_found = false;
        for (int j = 0; j < pose_num; ++j)
        {
            true_states.getPose(j, true_pose_se3.rotation(), true_pose_se3.translation());
            Vector3d point_cam = true_pose_se3.map(noise_point_i);
            z = cam.cam_map(point_cam);
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
            {
                pose_obs.push_back(std::pair<int, Vector2d>(j, z));
                if(ref_found==false){
                  inv_depth = 1.0/point_cam[2];
                  ref_found=true;
                }
            }
        }
        if (pose_obs.size() >= 2)
        {
            *(states.point(i)) = inv_depth+Utility::gaussian(0.2);
            ref_pose_obs[i]= pose_obs[0];
            problem.AddParameterBlock(states.point(i), 1);
            if(useOrdering)
                ordering->AddElementToGroup(states.point(i), 0);
            Vector2d ref_obs_noise = pose_obs[0].second +
                                     Vector2d(Utility::gaussian(PIXEL_NOISE),
                                                Utility::gaussian(PIXEL_NOISE));

            for(int j = 1; j < pose_obs.size(); ++j){
                Vector2d cur_obs_noise = pose_obs[j].second +
                                          Vector2d(Utility::gaussian(PIXEL_NOISE), Utility::gaussian(PIXEL_NOISE));
                ceres::CostFunction* costFunc = new InverseDepthFactor<PoseBlockSize>(focal_length, cx, cy, ref_obs_noise[0], ref_obs_noise[1],
                                                                          cur_obs_noise[0], cur_obs_noise[1]);
                //ceres::CostFunction* costFunc = new ceres::AutoDiffCostFunction< InverseDepthCostFunctor<PoseBlockSize>, 2, PoseBlockSize, PoseBlockSize, 1>(
                //                                          new InverseDepthCostFunctor<PoseBlockSize>(focal_length, cx, cy, ref_obs_noise[0], ref_obs_noise[1],
                //                                                          cur_obs_noise[0], cur_obs_noise[1]) );
                problem.AddResidualBlock(costFunc, NULL, states.pose(pose_obs[0].first), states.pose(pose_obs[j].first), states.point(i));
                double* param[] = {states.pose(pose_obs[0].first), states.pose(pose_obs[j].first), states.point(i)};
                double residual[2];
                costFunc->Evaluate(param, residual, 0);
                //std::cout << "residual: " << residual[0] << " " <<residual[1] << std::endl;
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
void BAProblemIDP<PoseBlockSize>::solve(ceres::Solver::Options& opt, ceres::Solver::Summary *sum)
{
    if(ordering != NULL)
        opt.linear_solver_ordering.reset(ordering);
    ceres::Solve(opt, &problem, sum);
}

template<int PoseBlockSize>
void BAProblemIDP<PoseBlockSize>::validate()
{
    double focal_length = 1000.;
    double cx = 320.;
    double cy = 240.;
    CameraParameters cam(focal_length, cx, cy);

    Eigen::Quaterniond Q;
    Eigen::Vector3d t;
    double cost = 0.0;
    for(size_t i = 0; i < ref_pose_obs.size(); ++i){
        if(ref_pose_obs[i].first == -1) continue;
        states.getPose(ref_pose_obs[i].first, Q, t);
        Eigen::Vector2d pix = ref_pose_obs[i].second;
        Eigen::Vector3d Cp = cam.cam_map_inv(pix, *(states.point(i)));
        Eigen::Vector3d Wp = Q.inverse()*(Cp - t);
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        cost += (Wp - true_point_i).norm() ;
    }
    std::cout << "Total 3d error: "<< cost << std::endl;
}
