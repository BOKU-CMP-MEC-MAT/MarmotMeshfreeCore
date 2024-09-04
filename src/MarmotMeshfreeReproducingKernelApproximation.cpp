
#include "Marmot/MarmotMeshfreeReproducingKernelApproximation.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

namespace Marmot::Meshfree {
  MarmotMeshfreeReproducingKernelApproximation::MarmotMeshfreeReproducingKernelApproximation( int dim,
                                                                                              int completenessOrder )
    : _dim( dim ), _completenessOrder( completenessOrder )
  {
  }

  constexpr int MarmotMeshfreeReproducingKernelApproximation::computeSizeHVector( int completenessOrder, int dim )
  {
    // compute the size of the H vector based on the completeness order and the dimension
    // for Eq. (3.67) in the book by Belytschko, Chen, Hillman.

    int size = 0;
    for ( int i = 0; i <= completenessOrder; i++ ) {
      if ( dim == 1 )
        size += 1;
      else
        size += computeSizeHVector( completenessOrder - i, dim - 1 );
    }
    return size;
  }

  int MarmotMeshfreeReproducingKernelApproximation::computeHRecursively( int                    completenessOrder,
                                                                         const Eigen::VectorXd& x_minus_xI,
                                                                         Eigen::VectorXd&       res,
                                                                         int                    idx,
                                                                         int                    dim )
  {
    for ( int i = 0; i <= completenessOrder; i++ ) {
      res( idx ) *= std::pow( x_minus_xI[dim - 1], i );
      if ( dim > 1 )
        idx = computeHRecursively( completenessOrder - i, x_minus_xI, res, idx, dim - 1 );
      else {
        idx++;
      }
    }
    return idx;
  }

  Eigen::VectorXd MarmotMeshfreeReproducingKernelApproximation::computeHVector(
    const Eigen::VectorXd&                                    x_minus_xI,
    const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
    const int                                                 completenessOrder )
  {
    const auto _sizeH = computeSizeHVector( completenessOrder, x_minus_xI.size() );

    Eigen::VectorXd res = Eigen::VectorXd::Ones( _sizeH );

    computeHRecursively( completenessOrder, x_minus_xI, res, 0, x_minus_xI.size() );

    /* std::cout << "H vector: " << res.transpose() << std::endl; */

    return res;
  }

  Eigen::VectorXd MarmotMeshfreeReproducingKernelApproximation::H0Vector( int sizeHVector )
  {
    Eigen::VectorXd H0 = Eigen::VectorXd::Zero( sizeHVector );
    H0( 0 )            = 1;
    return H0;
  }

  Eigen::MatrixXd MarmotMeshfreeReproducingKernelApproximation::computeMMatrix(
    const Eigen::VectorXd&                                    globalCoord,
    const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
    int                                                       completenessOrder )
  {
    /* std::cout << "Global coordinates: " << globalCoord << std::endl; */
    const auto _sizeH = computeSizeHVector( completenessOrder, globalCoord.size() );
    /* std::cout << "Size of H vector: " << _sizeH << std::endl; */

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero( _sizeH, _sizeH );

    for ( const auto& coveringShapeFunction : coveringShapeFunctions ) {
      const Eigen::Map< const Eigen::VectorXd > center( coveringShapeFunction->getCenterCoordinates(),
                                                        globalCoord.size() );

      const Eigen::VectorXd Hx = computeHVector( globalCoord - center, coveringShapeFunctions, completenessOrder );

      /* std::cout << "Hx: " << Hx << std::endl; */

      M += Hx * Hx.transpose() * coveringShapeFunction->computeKernelFunction( globalCoord.data() );
    }

    /* std::cout << "M: " << M << std::endl; */

    return M;
  }

  void MarmotMeshfreeReproducingKernelApproximation::computeShapeFunctions(
    const double*                                             coord,
    const std::vector< const MarmotMeshfreeKernelFunction* >& coveringKernelFunctions,
    double*                                                   shapeFunctionValues ) const
  {

    /* // MAp: */
    const Eigen::Map< const Eigen::VectorXd > coordVec( coord, _dim );

    const auto M = computeMMatrix( coordVec, coveringKernelFunctions, _completenessOrder );

    // solve for b(x)
    // b = M^-1 * H0
    const auto H0 = H0Vector( M.rows() );

    const Eigen::VectorXd b = M.colPivHouseholderQr().solve( H0 );

    // compute the shape function values

    for ( int i = 0; i < (int)coveringKernelFunctions.size(); i++ ) {
      const auto H = computeHVector( coordVec - Eigen::Map< const Eigen::VectorXd >( coveringKernelFunctions[i]
                                                                                       ->getCenterCoordinates(),
                                                                                     _dim ),
                                     coveringKernelFunctions,
                                     _completenessOrder );
      shapeFunctionValues[i] = b.dot( H ) * coveringKernelFunctions[i]->computeKernelFunction( coord );
    }
  }

  void MarmotMeshfreeReproducingKernelApproximation::computeShapeFunctionGradients(
    const double*                                             coord,
    const std::vector< const MarmotMeshfreeKernelFunction* >& kernelFunctions,
    double*                                                   shapeFunctionValueGradients ) const
  {
  }

  void MarmotMeshfreeReproducingKernelApproximation::computeShapeFunctionsAndGradients(
    const double*                                             coord,
    const std::vector< const MarmotMeshfreeKernelFunction* >& kernelFunctions,
    double*                                                   shapeFunctionValues,
    double*                                                   shapeFunctionValueGradients ) const
  {
  }

}; // namespace Marmot::Meshfree
