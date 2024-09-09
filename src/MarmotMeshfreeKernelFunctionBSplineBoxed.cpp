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

  void MarmotMeshfreeKernelFunctionBSplineBoxed::computeKernelFunctionGradient( const double* coord,
                                                                                double*       grad ) const
  {
    for ( int i = 0; i < _dim; i++ ) {
      grad[i] = 0;
    }

    for ( int i = 0; i < _dim; i++ ) {
      double res = 1.0;
      for ( int j = 0; j < _dim; j++ ) {
        if ( i == j )
          res *= computeBSpline2ndOrderGradient( coord[j] - _centerCoord[j] );
        else
          res *= computeBSpline2ndOrder( coord[j] - _centerCoord[j] );
      }
      grad[i] = res;
    }
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

  double MarmotMeshfreeKernelFunctionBSplineBoxed::computeBSpline2ndOrderGradient( double coord_minus_center ) const
  {
    const double z         = std::abs( coord_minus_center ) / _supportRadius;
    const double dz_dcoord = coord_minus_center > 0 ? 1.0 / _supportRadius : -1.0 / _supportRadius;

    if ( z <= 1. / 2 )
      return -4 * z * dz_dcoord;
    if ( z <= 1 )
      return ( -4 + 4 * z ) * dz_dcoord;
    return 0;
  }

  const double* MarmotMeshfreeKernelFunctionBSplineBoxed::getCenterCoordinates() const
  {
    return _centerCoord;
  }

  void MarmotMeshfreeKernelFunctionBSplineBoxed::moveTo( const double* coordinate )
  {
    for ( int i = 0; i < _dim; i++ ) {
      _centerCoord[i] = coordinate[i];
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
