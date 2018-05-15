#include "utility.h"



static double uniform_rand(double lowerBndr, double upperBndr)
{
    return lowerBndr + ((double)std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma)
{
    double x, y, r2;
    do
    {
        x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0.0);
    return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Utility::uniform(int from, int to)
{
    return static_cast<int>(uniform_rand(from, to));
}

double Utility::uniform()
{
    return uniform_rand(0., 1.);
}

double Utility::gaussian(double sigma)
{
    return gauss_rand(0., sigma);
}


