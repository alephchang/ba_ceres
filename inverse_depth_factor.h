#pragma once
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "se3.h"


template<int PoseBlockSize>
class InverseDepthFactor: public ceres::SizedCostFunction<2, 
                                PoseBlockSize,  
                                PoseBlockSize, 
                                1>
{
public:
    InverseDepthFactor(double f, double cx,  double cy,
              double ref_x, double ref_y,
              double obs_x, double obx_y)
        : f_(f), cx_(cx), cy_(cy),
          ref_(ref_x, ref_y),
          obs_(obs_x, obx_y){}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
    double f_;
    double cx_;
    double cy_;
    Eigen::Vector2d ref_;
    Eigen::Vector2d obs_;
};

template<int PoseBlockSize>
class InverseDepthCostFunctor{
public:
    InverseDepthCostFunctor(double f, double cx,  double cy,
              double ref_x, double ref_y,
              double obs_x, double obx_y)
        : f_(f), cx_(cx), cy_(cy),
          ref_(ref_x, ref_y),
          obs_(obs_x, obx_y){}
    template <typename T>
    bool operator()(const T* const p0 , const T* const p1, const T* const rho, T* e) const ;

    double f_;
    double cx_;
    double cy_;
    Eigen::Vector2d ref_;
    Eigen::Vector2d obs_;
};

template<>
template<typename T>
bool InverseDepthCostFunctor<7>::operator()(const T* const p0 , const T* const p1, const T* const inv_depth, T* e) const
{
    Eigen::Map<const Eigen::Quaternion<T> > Q_C0W(p0);
    Eigen::Map<const Eigen::Matrix<T,3, 1> > t_C0W(p0 + 4);
    Eigen::Map<const Eigen::Quaternion<T> > Q_C1W(p1);
    Eigen::Map<const Eigen::Matrix<T,3, 1> > t_C1W(p1 + 4);
    T rho(inv_depth[0]);
    Eigen::Matrix<T,3, 1> C0p((ref_[0]-cx_)/(f_*rho), (ref_[1]-cy_)/(f_*rho), 1.0/rho);
    Eigen::Matrix<T,3, 1> C0p_t_C0W = C0p - t_C0W;
    Eigen::Quaternion<T> Q_WC0 = Q_C0W.inverse();
    Eigen::Matrix<T,3, 1> Wp = Q_WC0*C0p_t_C0W; //i.e. Q_C0W.inverse()*(C0p - t_C0W)
    Eigen::Matrix<T,3, 1> C1p = Q_C1W*Wp + t_C1W;
    T f_by_z = f_ / C1p[2];
    e[0] = (f_by_z*C1p[0] + cx_) - obs_[0] ;
    e[1] = (f_by_z*C1p[1] + cy_) - obs_[1] ;

    return true;
}

template<>
template<typename T>
bool InverseDepthCostFunctor<6>::operator()(const T* const p0 , const T* const p1, const T* const inv_depth, T* e) const
{
    Eigen::Quaternion<T> Q_C0W = toQuaterniond(Eigen::Map<const Eigen::Matrix<T,3,1> >(p0));
    Eigen::Map<const Eigen::Matrix<T,3,1> > t_C0W(p0 + 3);
    Eigen::Quaterniond Q_C1W = toQuaterniond(Eigen::Map<const Eigen::Matrix<T,3,1> >(p1));
    Eigen::Map<const Eigen::Matrix<T,3,1> > t_C1W(p1 + 3);
    double rho(inv_depth[0]);
    Eigen::Matrix<T,3,1> C0p((ref_[0]-cx_)/(f_*rho), (ref_[1]-cy_)/(f_*rho), 1.0/rho);
    Eigen::Matrix<T,3,1> C0p_t_C0W = C0p - t_C0W;
    Eigen::Quaternion<T> Q_WC0 = Q_C0W.inverse();
    Eigen::Matrix<T,3,1> Wp = Q_WC0*C0p_t_C0W; //i.e. Q_C0W.inverse()*(C0p - t_C0W)

    Eigen::Matrix<T,3,1> C1p = Q_C1W*Wp + t_C1W;
    T f_by_z = f_ / C1p[2];
    e[0] = (f_by_z*C1p[0] + cx_) - obs_[0] ;
    e[1] = (f_by_z*C1p[1] + cy_) - obs_[1] ;

 
    return true;
}
