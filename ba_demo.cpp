#include "baproblem.h"
#include "baproblem_inverse_depth.h"
#include "baproblem_marginalization.h"

using namespace Eigen;
using namespace std;

const int USE_POSE_SIZE = 7;

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        cout << endl;
        cout << "Please type: " << endl;
        cout << "ba_demo [PIXEL_NOISE] " << endl;
        cout << endl;
        cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
        cout << endl;
        exit(0);
    }

    google::InitGoogleLogging(argv[0]);

    double PIXEL_NOISE = atof(argv[1]);

    cout << "PIXEL_NOISE: " << PIXEL_NOISE << endl;

    //bundle adjustment
    //BAProblem<USE_POSE_SIZE> baProblem(15, 300, PIXEL_NOISE); 
    //bundle adjustment with inverse depth model
    //BAProblemIDP<USE_POSE_SIZE> baProblem(15, 300, PIXEL_NOISE); 
    //bundle adjustment with inverse depth model and marginalization
    BAProblemMarg<USE_POSE_SIZE> baProblem(7, 1000, PIXEL_NOISE);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    baProblem.validate();
    baProblem.solveMarg(options, &summary);
    baProblem.validate();
    std::cout << summary.BriefReport() << "\n";
    return 0;
}
