
#include "Marmot/MarmotJournal.h"
#include "Marmot/MarmotMeshfreeReproducingKernelApproximation.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
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
                                                                         int                    idxEnd,
                                                                         int                    dim )
  {
    for ( int i = 0; i <= completenessOrder; i++ ) {

      const int idxStart = idxEnd;
      if ( dim > 1 )
        idxEnd = computeHRecursively( completenessOrder - i, x_minus_xI, res, idxEnd, dim - 1 );
      else {
        idxEnd++;
      }

      for ( int idx = idxStart; idx < idxEnd; idx++ )
        res( idx ) *= std::pow( x_minus_xI[dim - 1], i );
    }
    return idxEnd;
  }

  int MarmotMeshfreeReproducingKernelApproximation::computeHGradientRecursively( int completenessOrder,
                                                                                 const Eigen::VectorXd& x_minus_xI,
                                                                                 Eigen::MatrixXd&       res,
                                                                                 int                    idxEnd,
                                                                                 int                    dim )
  {
    for ( int i = 0; i <= completenessOrder; i++ ) {

      const int idxStart = idxEnd;

      if ( dim > 1 )
        idxEnd = computeHGradientRecursively( completenessOrder - i, x_minus_xI, res, idxEnd, dim - 1 );
      else {
        idxEnd++;
      }
      for ( int idx = idxStart; idx < idxEnd; idx++ ) {
        if ( i > 0 )
          res( idx, dim - 1 ) = i * std::pow( x_minus_xI[dim - 1], i - 1 );
        else
          res( idx, dim - 1 ) = 0;
      }
    }
    return idxEnd;
  }

  Eigen::VectorXd MarmotMeshfreeReproducingKernelApproximation::computeHVector(
    const Eigen::VectorXd&                                    x_minus_xI,
    const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
    const int                                                 completenessOrder )
  {
    const auto _sizeH = computeSizeHVector( completenessOrder, x_minus_xI.size() );

    Eigen::VectorXd res = Eigen::VectorXd::Ones( _sizeH );

    computeHRecursively( completenessOrder, x_minus_xI, res, 0, x_minus_xI.size() );

    return res;
  }

  Eigen::MatrixXd MarmotMeshfreeReproducingKernelApproximation::computeHVectorGradient(
    const Eigen::VectorXd&                                    x_minus_xI,
    const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
    const int                                                 completenessOrder )
  {
    const auto _sizeH = computeSizeHVector( completenessOrder, x_minus_xI.size() );

    Eigen::MatrixXd res = Eigen::MatrixXd::Ones( _sizeH, x_minus_xI.size() );

    computeHGradientRecursively( completenessOrder, x_minus_xI, res, 0, x_minus_xI.size() );

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
    const auto _sizeH = computeSizeHVector( completenessOrder, globalCoord.size() );

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero( _sizeH, _sizeH );

    for ( const auto& coveringShapeFunction : coveringShapeFunctions ) {
      const Eigen::Map< const Eigen::VectorXd > center( coveringShapeFunction->getCenterCoordinates(),
                                                        globalCoord.size() );

      const Eigen::VectorXd Hx = computeHVector( globalCoord - center, coveringShapeFunctions, completenessOrder );

      M += Hx * Hx.transpose() * coveringShapeFunction->computeKernelFunction( globalCoord.data() );
    }

    return M;
  }

  std::pair< Eigen::MatrixXd, std::vector< Eigen::MatrixXd > > MarmotMeshfreeReproducingKernelApproximation::
    computeMMatrixAndGradient( const Eigen::VectorXd&                                    globalCoord,
                               const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
                               int                                                       completenessOrder )
  {

    const int  _dim   = globalCoord.size();
    const auto _sizeH = computeSizeHVector( completenessOrder, _dim );

    Eigen::MatrixXd                M = Eigen::MatrixXd::Zero( _sizeH, _sizeH );
    std::vector< Eigen::MatrixXd > MGradients( _dim );
    for ( auto& MGradient : MGradients )
      MGradient = Eigen::MatrixXd::Zero( _sizeH, _sizeH );

    for ( const auto& coveringShapeFunction : coveringShapeFunctions ) {
      const Eigen::Map< const Eigen::VectorXd > center( coveringShapeFunction->getCenterCoordinates(),
                                                        globalCoord.size() );

      const Eigen::VectorXd Hx = computeHVector( globalCoord - center, coveringShapeFunctions, completenessOrder );
      const Eigen::MatrixXd HxGradient = computeHVectorGradient( globalCoord - center,
                                                                 coveringShapeFunctions,
                                                                 completenessOrder );

      const double    phi         = coveringShapeFunction->computeKernelFunction( globalCoord.data() );
      Eigen::VectorXd phiGradient = Eigen::VectorXd::Zero( _dim );
      coveringShapeFunction->computeKernelFunctionGradient( globalCoord.data(), phiGradient.data() );

      const Eigen::MatrixXd HxHT = Hx * Hx.transpose();
      M += HxHT * phi;

      for ( int i = 0; i < _dim; i++ ) {
        const Eigen::MatrixXd HGrad_i_xHT = HxGradient.col( i ) * Hx.transpose();
        MGradients[i] += ( HGrad_i_xHT + HGrad_i_xHT.transpose() ) * phi + HxHT * phiGradient( i );
      }
    }

    return { M, MGradients };
  }

  void MarmotMeshfreeReproducingKernelApproximation::computeShapeFunctions(
    const double*                                             coord,
    const std::vector< const MarmotMeshfreeKernelFunction* >& coveringKernelFunctions,
    double*                                                   shapeFunctionValues ) const
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

    const auto M = computeMMatrix( coordVec, coveringKernelFunctions, _completenessOrder );

    // solve for b(x)
    // b = M^-1 * H0
    const auto H0 = H0Vector( M.rows() );

    const Eigen::VectorXd b = M.colPivHouseholderQr().solve( H0 );

    // compute the shape function values

    for ( int A = 0; A < (int)coveringKernelFunctions.size(); A++ ) {
      const auto H = computeHVector( coordVec - Eigen::Map< const Eigen::VectorXd >( coveringKernelFunctions[A]
                                                                                       ->getCenterCoordinates(),
                                                                                     _dim ),
                                     coveringKernelFunctions,
                                     _completenessOrder );
      shapeFunctionValues[A] = b.dot( H ) * coveringKernelFunctions[A]->computeKernelFunction( coord );
    }
  }

  void MarmotMeshfreeReproducingKernelApproximation::computeShapeFunctionGradients(
    const double*                                             coord,
    const std::vector< const MarmotMeshfreeKernelFunction* >& kernelFunctions,
    double*                                                   shapeFunctionValueGradients ) const
  {
    throw std::runtime_error( "Not implemented" );
  }

  void MarmotMeshfreeReproducingKernelApproximation::computeShapeFunctionsAndGradients(
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

    const Eigen::Map< const Eigen::VectorXd > coordVec( coord, _dim );
    const auto                                sizeH = computeSizeHVector( _completenessOrder, _dim );

    Eigen::Map< Eigen::MatrixXd > shapeFunctionValueGradients( shapeFunctionValueGradients_,
                                                               _dim,
                                                               coveringKernelFunctions.size() );

    const auto [M, MGradients] = computeMMatrixAndGradient( coordVec, coveringKernelFunctions, _completenessOrder );

    // solve for b(x)
    // b = M^-1 * H0
    const auto H0 = H0Vector( sizeH );

    const auto MHr = M.colPivHouseholderQr();

    const Eigen::VectorXd b = MHr.solve( H0 );

    shapeFunctionValueGradients.setZero();

    for ( int A = 0; A < (int)coveringKernelFunctions.size(); A++ ) {

      const Eigen::VectorXd x_minus_center = coordVec - Eigen::Map< const Eigen::VectorXd >( coveringKernelFunctions[A]
                                                                                               ->getCenterCoordinates(),
                                                                                             _dim );

      const auto      phi_A         = coveringKernelFunctions[A]->computeKernelFunction( coord );
      Eigen::VectorXd phiGradient_A = Eigen::VectorXd::Zero( _dim );
      coveringKernelFunctions[A]->computeKernelFunctionGradient( coord, phiGradient_A.data() );

      const auto H         = computeHVector( x_minus_center, coveringKernelFunctions, _completenessOrder );
      const auto HGradient = computeHVectorGradient( x_minus_center, coveringKernelFunctions, _completenessOrder );

      shapeFunctionValues[A] = b.dot( H ) * phi_A;

      const Eigen::MatrixXd MInv_HGrad = MHr.solve( HGradient );
      const Eigen::VectorXd MInv_H     = MHr.solve( H );

      // let's compute the gradient of the shape function
      // dPsiA_dxi = H0_J * ( InvM_JK * H_K * phi_A ),xi
      //
      // dPsiA_dxi = H0_J * ( InvM_JK,xi * H_K    * phi_A +
      //                      InvM_JK    * H_K,xi * phi_A +
      //                      InvM_JK    * H_K    * phi_A,xi )
      //
      // dPsiA_dxi = H0_J * ( - [InvM_JA * M_AB,xi * InvM_BK]    * H_K    * phi_A +
      //                      InvM_JK                            * H_K,xi * phi_A +
      //                      InvM_JK                            * H_K    * phi_A,xi )

      // we exploit the fact that H0 = 1 for idx = 0, and 0 otherwise
      // so really we never need to compute the dot product with H0, which involves a lot of zeros

      shapeFunctionValueGradients.col( A ) += MInv_HGrad.row( 0 ) * phi_A;
      shapeFunctionValueGradients.col( A ) += MInv_H( 0 ) * phiGradient_A;
      for ( int i = 0; i < _dim; i++ ) {
        shapeFunctionValueGradients( i, A ) += ( -MHr.solve( MGradients[i] ) * MInv_H )( 0 ) * phi_A;
      }
    }
  }

}; // namespace Marmot::Meshfree
