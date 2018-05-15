#pragma once

#include "inverse_depth_factor.h"
#include "marginalization_factor.h"
#include "utility.h"

template<int PoseBlockSize>
class BAProblemMarg
{
public:
    BAProblemMarg(int pose_num_, int point_num_, double pix_noise_);

    void solve(ceres::Solver::Options &opt, ceres::Solver::Summary* sum);
    void solveMarg(ceres::Solver::Options &opt, ceres::Solver::Summary* sum);

    void validate();

    ceres::Problem problem; //a global BA prolem solver, for test

protected:
    PosePointParametersBlock<PoseBlockSize> true_states;

    PosePointIDPParametersBlock<PoseBlockSize> states; // states.point(i) is inv_depth, 1 dimension
    std::vector<std::vector<int> > ref_pose_pts; //size is pose_num, the ith vector store the point ids the ith frame see
    std::vector< std::vector<std::pair<int, Vector2d> > > pt_pose_obs; //size is point_num, the ith vector stores frame id and the obs about the ith point, the first frame is chosen as reference(host) frame
};

template<int PoseBlockSize>
BAProblemMarg<PoseBlockSize>::BAProblemMarg(int pose_num_, int point_num_, double pix_noise_)
{
    int pose_num = pose_num_;
    int point_num = point_num_;
    double PIXEL_NOISE = pix_noise_;

    states.create(pose_num, point_num);
    true_states.create(pose_num, point_num);

    double focal_length = 1000.;
    double cx = 320.;
    double cy = 240.;
    CameraParameters cam(focal_length, cx, cy);

    //create the cameras
    std::vector<double> pose_xs(pose_num);
    for (int i = 0; i < pose_num; ++i)
    {
        pose_xs[i] = i * 0.5 - 1.;
        Vector3d trans(pose_xs[i], 0.0, 0.0);

        Eigen::Quaterniond q;
        q.setIdentity();
        true_states.setPose(i, q, trans);
        trans[0] += Utility::gaussian(0.1);
        states.setPose(i, q, trans);

        problem.AddParameterBlock(states.pose(i), PoseBlockSize, new PoseSE3Parameterization<PoseBlockSize>());

        if(i < 1)
        {
            problem.SetParameterBlockConstant(states.pose(i));
        }
    }

    int step = point_num/pose_num;
    for(int icam = 0; icam < pose_num; ++ icam){
        for(int i = 0; i < step; ++i){
            Eigen::Map<Vector3d> true_pt(true_states.point(icam*step+i));
            true_pt = Vector3d((Utility::uniform() - 0.5) * 3 - pose_xs[icam],
                           Utility::uniform() - 0.5,
                           Utility::uniform() + 3);
        }
    }
    for(int i = step*pose_num; i < point_num; ++ i){
         Eigen::Map<Vector3d> true_pt(true_states.point(i));
         true_pt = Vector3d((Utility::uniform() - 0.5) * 3 + pose_xs.back(),
                           Utility::uniform() - 0.5,
                           Utility::uniform() + 3);
    }

    ref_pose_pts.resize(pose_num);
    pt_pose_obs.resize(point_num);
    for (int i = 0; i < point_num; ++i)
    {
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        Eigen::Vector3d noise_point_i;
        noise_point_i = true_point_i + Vector3d(Utility::gaussian(0.3),
                                                Utility::gaussian(0.3),
                                                Utility::gaussian(0.3));

        Vector2d z;
        SE3 true_pose_se3;
        double inv_depth;
        bool ref_found = false;
        for (int j = 0; j < pose_num; ++j)
        {
            true_states.getPose(j, true_pose_se3.rotation(), true_pose_se3.translation());
            Vector3d point_cam = true_pose_se3.map(noise_point_i);
            z = cam.cam_map(point_cam) + 
                            Vector2d(Utility::gaussian(PIXEL_NOISE),
                                     Utility::gaussian(PIXEL_NOISE));
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
            {
                pt_pose_obs[i].push_back(std::pair<int, Vector2d>(j, z));
                if(ref_found==false){
                  inv_depth = 1.0/point_cam[2];
                  ref_found=true;
                }
            }
        }
        if (pt_pose_obs[i].size() >= 2)
        {
            *(states.point(i)) = inv_depth+Utility::gaussian(0.2);
            ref_pose_pts[pt_pose_obs[i][0].first].push_back(i);
            problem.AddParameterBlock(states.point(i), 1);
            Vector2d ref_obs_noise = pt_pose_obs[i][0].second; 

            for(int j = 1; j < pt_pose_obs[i].size(); ++j){
                Vector2d cur_obs_noise = pt_pose_obs[i][j].second;
                ceres::CostFunction* costFunc = new InverseDepthFactor<PoseBlockSize>(focal_length, cx, cy, ref_obs_noise[0], ref_obs_noise[1],
                                                                          cur_obs_noise[0], cur_obs_noise[1]);
                //ceres::CostFunction* costFunc = new ceres::AutoDiffCostFunction< IDPCostFunctor<PoseBlockSize>, 2, PoseBlockSize, PoseBlockSize, 1>(
                //                                          new IDPCostFunctor<PoseBlockSize>(focal_length, cx, cy, ref_obs_noise[0], ref_obs_noise[1],
                //                                                          cur_obs_noise[0], cur_obs_noise[1]) );
                problem.AddResidualBlock(costFunc, NULL, states.pose(pt_pose_obs[i][0].first), states.pose(pt_pose_obs[i][j].first), states.point(i));
                double* param[] = {states.pose(pt_pose_obs[i][0].first), states.pose(pt_pose_obs[i][j].first), states.point(i)};
                double residual[2];
                costFunc->Evaluate(param, residual, 0);
                //std::cout << "residual: " << residual[0] << " " <<residual[1] << std::endl;
            }
        }
    }
}

template<int PoseBlockSize>
void BAProblemMarg<PoseBlockSize>::solve(ceres::Solver::Options &opt, ceres::Solver::Summary* sum)
{
    ceres::Solve(opt, &problem, sum);
}

template<int PoseBlockSize>
void BAProblemMarg<PoseBlockSize>::solveMarg(ceres::Solver::Options &opt, ceres::Solver::Summary* sum)
{
    ceres::Problem problem0;
    const int WINDOW_SIZE = 5;
    double focal_length = 1000.;
    double cx = 320.;
    double cy = 240.;
    CameraParameters cam(focal_length, cx, cy);
    //1. BA for the first WINDOW_SIZE poses
    //1.1 create the cameras
    for(size_t i = 0; i < WINDOW_SIZE; ++i){
        problem0.AddParameterBlock(states.pose(i), PoseBlockSize, new PoseSE3Parameterization<PoseBlockSize>());
        if(i<1)
            problem0.SetParameterBlockConstant(states.pose(i));
    }
    //1.2 add the cost function, i.e. edge
    for(size_t i = 0; i < states.pointNum; ++i){
        const std::vector<std::pair<int, Vector2d> >& obs = pt_pose_obs[i];
        if(obs.empty()) continue;
        if(obs[0].first >= WINDOW_SIZE) break;
        if(obs.size()<2) continue;
        if(obs[1].first >= WINDOW_SIZE) continue;
        problem0.AddParameterBlock(states.point(i), 1);
        Vector2d ref_obs = obs[0].second;
        for(size_t j = 1; j < obs.size(); ++j){
            if(obs[j].first>=WINDOW_SIZE) break;
            Vector2d cur_obs = obs[j].second;
            ceres::CostFunction* costFunc = new InverseDepthFactor<PoseBlockSize>(focal_length, cx, cy, ref_obs[0], ref_obs[1],
                                                                          cur_obs[0], cur_obs[1]);
            problem0.AddResidualBlock(costFunc, NULL, states.pose(obs[0].first), states.pose(obs[j].first), states.point(i));
        }
    }
    ceres::Solve(opt, &problem0, sum);
    
    //loop to solve
    MarginalizationInfo *last_marginalization_info = 0;
    std::vector<double*> last_marginalization_parameter_blocks;
    for(size_t ipose = 1 ; ipose < states.poseNum-WINDOW_SIZE+1; ++ipose){
        //2.slide the window and marginalization
        //2.1 create marginalization
        //the pose ipose-1 should be marged.
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        int margId = ipose - 1;
        if(last_marginalization_info != 0){
            std::vector<int> drop_set;
            for(size_t i = 0; i < last_marginalization_parameter_blocks.size(); ++i){
                if(last_marginalization_parameter_blocks[i] == states.pose(margId)){
                    drop_set.push_back(i);
                }
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

           marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        const std::vector<int>& ref_pts = ref_pose_pts[margId];
        std::set<int> margPts;
        for(size_t i = 0; i < ref_pts.size(); ++i){
            const std::vector<std::pair<int, Vector2d> >& obs = pt_pose_obs[ref_pts[i]];
            if(obs.empty()) continue;
            assert( obs[0].first == margId );
            if(obs.size() < 2 ) continue;
            if(obs[i].first >= ipose+WINDOW_SIZE) continue;
            Vector2d ref_obs = obs[0].second;
            std::vector<int> drop_set;
            drop_set.push_back(0); drop_set.push_back(2);
            std::vector<double*> param(3);
            for(size_t j = 1; j < obs.size(); ++j){
                if(obs[j].first >= ipose+WINDOW_SIZE) break;
                Vector2d cur_obs = obs[j].second;
                ceres::CostFunction* costFunc = new InverseDepthFactor<PoseBlockSize>(focal_length, cx, cy, ref_obs[0], ref_obs[1],
                                                                          cur_obs[0], cur_obs[1]);
                param[0] = states.pose(obs[0].first); param[1] = states.pose(obs[j].first); param[2] = states.point(ref_pts[i]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(costFunc, NULL, param, drop_set);
                marginalization_info->addResidualBlockInfo(residual_block_info);
                //std::cout <<"redisual_block_info: " << obs[0].first << " " << obs[j].first << " " << ref_pts[i] << std::endl;
                margPts.insert(ref_pts[i]);
            }
        }
        marginalization_info->preMarginalize();
        std::cout << ipose << std::endl;
        marginalization_info->marginalize();
        ceres::Problem problemMargi;
       // construct new marginlization_factor
        std::vector<double* > marginalization_parameter_blocks= marginalization_info->getParameterBlocks();
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(marginalization_info);
        //2.2 add the marginalization
        problemMargi.AddResidualBlock(marginalization_factor, NULL, marginalization_parameter_blocks);
        

        //2.3 create the cameras
        for(size_t i = ipose; i < ipose+WINDOW_SIZE; ++i){
            problemMargi.AddParameterBlock(states.pose(i), PoseBlockSize, new PoseSE3Parameterization<PoseBlockSize>());
        }

        //2.3 add the edges
        for(size_t i = 0; i < states.pointNum; ++i){
            const std::vector<std::pair<int, Vector2d> >& obs = pt_pose_obs[i];
            if(obs.empty()) continue;
            if(obs[0].first >= ipose+WINDOW_SIZE) break;
            if(obs.size()<2) continue;
            if(obs[1].first >= ipose+ WINDOW_SIZE) continue;
            problemMargi.AddParameterBlock(states.point(i), 1);
            Vector2d ref_obs = obs[0].second;
            for(size_t j = 1; j < obs.size(); ++j){
                if(obs[j].first>=ipose+WINDOW_SIZE) break;
                Vector2d cur_obs = obs[j].second;
                ceres::CostFunction* costFunc = new InverseDepthFactor<PoseBlockSize>(focal_length, cx, cy, ref_obs[0], ref_obs[1],
                                                                          cur_obs[0], cur_obs[1]);
                problemMargi.AddResidualBlock(costFunc, NULL, states.pose(obs[0].first), states.pose(obs[j].first), states.point(i));
            }
        }
        ceres::Solve(opt, &problemMargi, sum);
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = marginalization_parameter_blocks;
    }
}

template<int PoseBlockSize>
void BAProblemMarg<PoseBlockSize>::validate()
{
    double focal_length = 1000.;
    double cx = 320.;
    double cy = 240.;
    CameraParameters cam(focal_length, cx, cy);

    Eigen::Quaterniond Q;
    Eigen::Vector3d t;
    double cost = 0.0;
    for(size_t i = 0; i < pt_pose_obs.size(); ++i){
        if(pt_pose_obs[i].size()<2) continue;
        if(pt_pose_obs[i][0].first > 25) continue;
        states.getPose(pt_pose_obs[i][0].first, Q, t);
        Eigen::Vector2d pix = pt_pose_obs[i][0].second;
        Eigen::Vector3d Cp = cam.cam_map_inv(pix, *(states.point(i)));
        Eigen::Vector3d Wp = Q.inverse()*(Cp - t);
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        cost += (Wp - true_point_i).norm() ;
    }
    double pose_cost = 0.0;
    for(size_t i = 0; i < states.poseNum; ++i){
        Eigen::Map<Vector3d> true_trans(true_states.pose(i)+(PoseBlockSize==7? 4 : 3) );
        Eigen::Map<Vector3d> esti_trans(states.pose(i)+(PoseBlockSize==7? 4 : 3) );
        pose_cost += (esti_trans - true_trans).norm();
    }
    std::cout << "Total 3d error: "<< cost << " pose error: " << pose_cost <<std::endl;
}
