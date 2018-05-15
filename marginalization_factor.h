#pragma once

#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <map>



/*
 * Used to store any kinds of factors information
 *
 */
struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function,
                      ceres::LossFunction *_loss_function,
                      std::vector<double *> _parameter_blocks,
                      std::vector<int> _drop_set)
        : cost_function(_cost_function),
          loss_function(_loss_function),
          parameter_blocks(_parameter_blocks), //
          drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::map<long, int> parameter_block_size; //global size
    std::map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks();

    std::vector<ResidualBlockInfo *> factors; // factors involing to parametre need be marginalized
    int m, n;
    std::map<long, int> parameter_block_size; //global size, map address to size
    int sum_block_size;
    std::map<long, int> parameter_block_idx; //local size, map address
    std::map<long, double *> parameter_block_data; // map address

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians; // H
    Eigen::VectorXd linearized_residuals; // b
    static double eps;// = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

    MarginalizationInfo* marginalization_info;
};
