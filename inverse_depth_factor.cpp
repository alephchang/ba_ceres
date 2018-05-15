#include"inverse_depth_factor.h"

template<>
bool InverseDepthFactor<7>::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Eigen::Quaterniond> Q_C0W(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t_C0W(parameters[0] + 4);
    Eigen::Map<const Eigen::Quaterniond> Q_C1W(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> t_C1W(parameters[1] + 4);
    double rho(parameters[2][0]);
    Eigen::Vector3d C0p((ref_[0]-cx_)/(f_*rho), (ref_[1]-cy_)/(f_*rho), 1.0/rho);
    Eigen::Vector3d C0p_t_C0W = C0p - t_C0W;
    Eigen::Quaterniond Q_WC0 = Q_C0W.inverse();
    Eigen::Vector3d Wp = Q_WC0*C0p_t_C0W; //i.e. Q_C0W.inverse()*(C0p - t_C0W)
    Eigen::Vector3d C1p = Q_C1W*Wp + t_C1W;
    double f_by_z = f_ / C1p[2];
    residuals[0] = (f_by_z*C1p[0] + cx_) - obs_[0] ;
    residuals[1] = (f_by_z*C1p[1] + cy_) - obs_[1] ;
    if(jacobians!=0){
        Eigen::Matrix3d R_C1W = Q_C1W.toRotationMatrix();
        Eigen::Matrix3d R_WC0 = Q_WC0.toRotationMatrix();
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
        double f_by_zz = f_by_z / C1p[2];
        J_cam << f_by_z, 0, - f_by_zz * C1p[0],
                0, f_by_z, - f_by_zz * C1p[1];
        if(jacobians[0]!=0){
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();
            J_se3.block<2,3>(0,0) = J_cam * R_C1W*R_WC0*skew(C0p_t_C0W);
            J_se3.block<2,3>(0,3) = -J_cam* R_C1W * R_WC0;
        }
        if(jacobians[1]!=0){
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3(jacobians[1]);
            J_se3.setZero();
            J_se3.block<2,3>(0,0) = - J_cam * skew(C1p);
            J_se3.block<2,3>(0,3) = J_cam;
        }
        if(jacobians[2]!=0){
            Eigen::Map<Eigen::Vector2d> J_rho(jacobians[2]);
            Eigen::Matrix3d J_C1p_C0p = R_C1W * R_WC0;
            Eigen::Matrix<double, 3, 1> J_C0p_rho= -C0p/rho;
            J_rho = J_cam * J_C1p_C0p * J_C0p_rho;
        }
    }
//    std::cout << "eval jacob" << std::endl;
    return true;
}

template<>
bool InverseDepthFactor<6>::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Quaterniond Q_C0W = toQuaterniond(Eigen::Map<const Eigen::Vector3d>(parameters[0]));
    Eigen::Map<const Eigen::Vector3d> t_C0W(parameters[0] + 3);
    Eigen::Quaterniond Q_C1W = toQuaterniond(Eigen::Map<const Eigen::Vector3d>(parameters[1]));
    Eigen::Map<const Eigen::Vector3d> t_C1W(parameters[1] + 3);
    double rho(parameters[2][0]);
    Eigen::Vector3d C0p((ref_[0]-cx_)/(f_*rho), (ref_[1]-cy_)/(f_*rho), 1.0/rho);
    Eigen::Vector3d C0p_t_C0W = C0p - t_C0W;
    Eigen::Quaterniond Q_WC0 = Q_C0W.inverse();
    Eigen::Vector3d Wp = Q_WC0*C0p_t_C0W; //i.e. Q_C0W.inverse()*(C0p - t_C0W)

    Eigen::Vector3d C1p = Q_C1W*Wp + t_C1W;
    double f_by_z = f_ / C1p[2];
    residuals[0] = (f_by_z*C1p[0] + cx_) - obs_[0] ;
    residuals[1] = (f_by_z*C1p[1] + cy_) - obs_[1] ;
    if(jacobians!=0){
        Eigen::Matrix3d R_C1W = Q_C1W.toRotationMatrix();
        Eigen::Matrix3d R_WC0 = Q_WC0.toRotationMatrix();
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
        double f_by_zz = f_by_z / C1p[2];
        J_cam << f_by_z, 0, - f_by_zz * C1p[0],
                0, f_by_z, - f_by_zz * C1p[1];
        if(jacobians[0]!=0){
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.block<2,3>(0,0) = J_cam * R_C1W*R_WC0*skew(C0p_t_C0W);
            J_se3.block<2,3>(0,3) = -J_cam* R_C1W * R_WC0;
        }
        if(jacobians[1]!=0){
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > J_se3(jacobians[1]);
            J_se3.block<2,3>(0,0) = - J_cam * skew(C1p);
            J_se3.block<2,3>(0,3) = J_cam;
        }
        if(jacobians[2]!=0){
            Eigen::Map<Eigen::Vector2d> J_rho(jacobians[2]);
            Eigen::Matrix3d J_C1p_C0p = R_C1W * R_WC0;
            Eigen::Matrix<double, 3, 1> J_C0p_rho= -C0p/rho;// ((ref_[0]-cx_)/f, (ref_[1]-cy)/f_, 1.0);
            J_rho = J_cam * J_C1p_C0p * J_C0p_rho;
        }
    }
    return true;
}
/*
template<>
template<typename T>
bool IDPCostFunctor<7>::operator()(const T* const p0 , const T* const p1, const T* const inv_depth, T* e) const
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
bool IDPCostFunctor<6>::operator()(const T* const p0 , const T* const p1, const T* const inv_depth, T* e) const
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
*/
