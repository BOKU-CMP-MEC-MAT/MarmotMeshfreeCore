#include "Marmot/MarmotMeshfreeKernelFunctionBSplineBoxed.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

namespace Marmot::Meshfree {

  MarmotMeshfreeKernelFunctionBSplineBoxed::MarmotMeshfreeKernelFunctionBSplineBoxed( double* centerCoord,
                                                                                      int     dim,
                                                                                      double  supportRadius,
                                                                                      int     continuityOrder )
    : _centerCoord( centerCoord ), _supportRadius( supportRadius ), _dim( dim ), _continuityOrder( continuityOrder )
  {
  }

  double MarmotMeshfreeKernelFunctionBSplineBoxed::computeKernelFunction( const double* coord ) const
  {

    double res = 1.0;

    for ( int i = 0; i < _dim; i++ ) {
      res *= computeBSpline2ndOrder( coord[i] - _centerCoord[i] );
    }

    return res;
  }

  double MarmotMeshfreeKernelFunctionBSplineBoxed::computeBSpline2ndOrder( double coord_minus_center ) const
  {
    const double z = std::abs( coord_minus_center ) / _supportRadius;

    if ( z <= 1. / 2 )
      return 1 - 2 * z * z;
    if ( z <= 1 )
      return 2 - 4 * z + 2 * z * z;
    return 0;
  }

  const double* MarmotMeshfreeKernelFunctionBSplineBoxed::getCenterCoordinates() const
  {
    return _centerCoord;
  }

  void MarmotMeshfreeKernelFunctionBSplineBoxed::move( const double* centerDisplacement )
  {
    for ( int i = 0; i < _dim; i++ ) {
      _centerCoord[i] += centerDisplacement[i];
    }
  }

  bool MarmotMeshfreeKernelFunctionBSplineBoxed::isInSupport( const double* coord ) const
  {
    return computeKernelFunction( coord ) > 0;
  }

  void MarmotMeshfreeKernelFunctionBSplineBoxed::getBoundingBox( double* min, double* max ) const
  {
    for ( int i = 0; i < _dim; i++ ) {
      min[i] = _centerCoord[i] - _supportRadius;
      max[i] = _centerCoord[i] + _supportRadius;
    }
  }

}; // namespace Marmot::Meshfree
