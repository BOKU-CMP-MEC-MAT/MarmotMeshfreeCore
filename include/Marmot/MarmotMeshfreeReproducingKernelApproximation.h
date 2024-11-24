/* ---------------------------------------------------------------------
 *                                       _
 *  _ __ ___   __ _ _ __ _ __ ___   ___ | |_
 * | '_ ` _ \ / _` | '__| '_ ` _ \ / _ \| __|
 * | | | | | | (_| | |  | | | | | | (_) | |_
 * |_| |_| |_|\__,_|_|  |_| |_| |_|\___/ \__|
 *
 * Unit of Strength of Materials and Structural Analysis
 * University of Innsbruck,
 * 2020 - today
 *
 * festigkeitslehre@uibk.ac.at
 *
 * Matthias Neuner matthias.neuner@uibk.ac.at
 *
 * This file is part of the MAteRialMOdellingToolbox (marmot).
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of marmot.
 * ---------------------------------------------------------------------
 */

#pragma once

#include "Marmot/MarmotMeshfreeApproximation.h"
#include "Marmot/MarmotMeshfreeKernelFunction.h"
#include <Eigen/Core>
#include <cmath>
#include <vector>

namespace Marmot::Meshfree {

  class MarmotMeshfreeReproducingKernelApproximation : public MarmotMeshfreeApproximation {

  protected:
    int _dim;
    int _completenessOrder;

    int static computeHRecursively( int                    completenessOrder,
                                    const Eigen::VectorXd& x_minus_center,
                                    Eigen::VectorXd&       res,
                                    int                    idx,
                                    int                    dim );

    int static computeHGradientRecursively( int                    completenessOrder,
                                            const Eigen::VectorXd& x_minus_center,
                                            Eigen::MatrixXd&       res,
                                            int                    idx,
                                            int                    dim );

    static constexpr int computeSizeHVector( int completenessOrder, int dim );

    static Eigen::VectorXd computeHVector(
      const Eigen::VectorXd&                                    globalCoord,
      const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
      int                                                       completenessOrder );

    static Eigen::MatrixXd computeHVectorGradient(
      const Eigen::VectorXd&                                    globalCoord,
      const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
      int                                                       completenessOrder );

    static Eigen::MatrixXd computeMMatrix(
      const Eigen::VectorXd&                                    globalCoord,
      const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
      int                                                       completenessOrder );

    static std::pair< Eigen::MatrixXd, std::vector< Eigen::MatrixXd > > computeMMatrixAndGradient(
      const Eigen::VectorXd&                                    globalCoord,
      const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
      int                                                       completenessOrder );

    static Eigen::VectorXd H0Vector( int sizeHVector );

    int factorial( int n ) const
    {
      if ( n == 0 )
        return 1;
      else
        return n * factorial( n - 1 );
    }

    bool checkNonSingularity( int nNodes ) const
    {
      if ( _completenessOrder > 0 ) {
        /* std::cout << nNodes << " >= " << factorial(_dim +
         * _completenessOrder)/(factorial(_dim)*factorial(_completenessOrder)) << std::endl; */
        return nNodes >=
               factorial( _dim + _completenessOrder ) / ( factorial( _dim ) * factorial( _completenessOrder ) );
      }
      else
        return true;
    }

    std::tuple<double, const std::vector< const MarmotMeshfreeKernelFunction* >, std::vector< int >  > findCoveringKernelFunctionsAndCompletenessOrder( const double* coord, const std::vector< const MarmotMeshfreeKernelFunction* >& kernelFunctions ) const;

  public:
    MarmotMeshfreeReproducingKernelApproximation( int dim, int completenessOrder );

    virtual ~MarmotMeshfreeReproducingKernelApproximation() = default;

    virtual void computeShapeFunctions(
      const double*                                             coord,
      const std::vector< const MarmotMeshfreeKernelFunction* >& coveringShapeFunctions,
      double*                                                   shapeFunctionValues ) const override;

    virtual void computeShapeFunctionGradients(
      const double*                                             coord,
      const std::vector< const MarmotMeshfreeKernelFunction* >& kernelFunctions,
      double*                                                   shapeFunctionValueGradients ) const override;

    virtual void computeShapeFunctionsAndGradients(
      const double*                                             coord,
      const std::vector< const MarmotMeshfreeKernelFunction* >& kernelFunctions,
      double*                                                   shapeFunctionValues,
      double*                                                   shapeFunctionValueGradients_ColMajor ) const override;
  };

} // namespace Marmot::Meshfree
