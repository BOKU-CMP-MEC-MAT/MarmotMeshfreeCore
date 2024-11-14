
#include "Marmot/MarmotJournal.h"
#include "Marmot/MarmotMeshfreeReproducingKernelApproximationImplicit.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cmath>

namespace Marmot::Meshfree {
  MarmotMeshfreeReproducingKernelApproximationImplicit::MarmotMeshfreeReproducingKernelApproximationImplicit(
    int dim,
    int completenessOrder )
    : MarmotMeshfreeReproducingKernelApproximation( dim, completenessOrder )
  {
  }

  void MarmotMeshfreeReproducingKernelApproximationImplicit::computeShapeFunctionGradients(
    const double*                                             coord,
    const std::vector< const MarmotMeshfreeKernelFunction* >& kernelFunctions,
    double*                                                   shapeFunctionValueGradients ) const
  {
    throw std::runtime_error( "Not implemented" );
  }

  void MarmotMeshfreeReproducingKernelApproximationImplicit::computeShapeFunctionsAndGradients(
    const double*                                             coord,
    const std::vector< const MarmotMeshfreeKernelFunction* >& coveringKernelFunctions,
    double*                                                   shapeFunctionValues,
    double*                                                   shapeFunctionValueGradients_ ) const
  {
    if ( !checkNonSingularity( coveringKernelFunctions.size() ) )
      throw std::runtime_error( MakeString() << __PRETTY_FUNCTION__
                                             << " : Not enough nodes to compute the shape functions due to singularity "
                                                "equation system of the correction:"
                                             << coveringKernelFunctions.size() << " < "
                                             << factorial( _dim + _completenessOrder ) /
                                                  ( factorial( _dim ) * factorial( _completenessOrder ) ) );

    /* // MAp: */
    const Eigen::Map< const Eigen::VectorXd > coordVec( coord, _dim );

    Eigen::Map< Eigen::MatrixXd > shapeFunctionValueGradients( shapeFunctionValueGradients_,
                                                               _dim,
                                                               coveringKernelFunctions.size() );

    const auto M = computeMMatrix( coordVec, coveringKernelFunctions, _completenessOrder );

    // solve for b(x)
    // b = M^-1 * H0
    const auto H0 = H0Vector( M.rows() );

    using namespace Eigen; 

    // first column is for the base function
    // the rest of the columns are for the gradients
    MatrixXd H0Mat( M.rows(), 1 + _dim );
    H0Mat.setZero();

    H0Mat.col(0) = H0;
    H0Mat.block(1,1, _dim, _dim).diagonal().setConstant( -1.0 );

    const Eigen::MatrixXd bMat = M.colPivHouseholderQr().solve( H0Mat );

    const VectorXd b0 = bMat.col(0);
    const MatrixXd bGrad = bMat.block(0, 1, M.rows(), _dim);

    for ( int A = 0; A < (int)coveringKernelFunctions.size(); A++ ) {
      const auto H = computeHVector( coordVec - Eigen::Map< const Eigen::VectorXd >( coveringKernelFunctions[A]
                                                                                       ->getCenterCoordinates(),
                                                                                     _dim ),
                                     coveringKernelFunctions,
                                     _completenessOrder );

      shapeFunctionValues[A] = b0.dot( H ) * coveringKernelFunctions[A]->computeKernelFunction( coord );

      shapeFunctionValueGradients.col( A ) = bGrad.transpose() * H * coveringKernelFunctions[A]->computeKernelFunction( coord );

    }
  }

}; // namespace Marmot::Meshfree
