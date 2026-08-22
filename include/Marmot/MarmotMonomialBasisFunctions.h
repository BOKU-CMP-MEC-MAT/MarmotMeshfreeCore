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

#include <Eigen/Core>
#include <cmath>

namespace Marmot::Math {

  /**
   * @brief @p base raised to a small non-negative integer power, by repeated multiplication.
   *
   * The monomial basis only ever needs exponents up to the completeness order, which is one or two
   * in practice. But the exponent is a loop variable rather than a compile-time constant, so the
   * compiler cannot turn std::pow into multiplications and every call goes to the library -- inside
   * the innermost loop of the reproducing-kernel reconstruction, to compute values as trivial as
   * 1 and x.
   *
   * For the exponents that occur here the result is bit-for-bit what std::pow returns: pow(x, 0) is
   * exactly 1, pow(x, 1) is exactly x, and pow(x, 2) rounds to the same double as x * x. From
   * exponent three upwards repeated multiplication rounds more than once and the last bit may
   * differ, which matters only for a completeness order of three or above.
   *
   * @param base The base.
   * @param exponent The non-negative exponent.
   * @return @p base raised to @p exponent.
   */
  inline double integerPower( double base, int exponent )
  {
    double result = 1.0;

    for ( int i = 0; i < exponent; i++ )
      result *= base;

    return result;
  }

  inline int computeSizeOfMonomialBasisVector( int order, int dim )
  {
    // compute the size of the H vector based on the completeness order and the dimension
    // for Eq. (3.67) in the book by Belytschko, Chen, Hillman.

    int size = 0;
    for ( int i = 0; i <= order; i++ ) {
      if ( dim == 1 )
        size += 1;
      else
        size += computeSizeOfMonomialBasisVector( order - i, dim - 1 );
    }
    return size;
  }

  inline int _computeMonomialBasisRecursion( int                    order,
                                             const Eigen::VectorXd& x,
                                             Eigen::VectorXd&       res,
                                             int                    idxEnd,
                                             int                    dim )
  {
    for ( int i = 0; i <= order; i++ ) {

      const int idxStart = idxEnd;
      if ( dim > 1 )
        idxEnd = _computeMonomialBasisRecursion( order - i, x, res, idxEnd, dim - 1 );
      else {
        idxEnd++;
      }

      for ( int idx = idxStart; idx < idxEnd; idx++ )
        res( idx ) *= integerPower( x[dim - 1], i );
    }
    return idxEnd;
  }

  inline int _computeMonomialBasisGradientRecursion( int                    order,
                                                     const Eigen::VectorXd& x,
                                                     Eigen::MatrixXd&       res,
                                                     int                    idxEnd,
                                                     int                    dim )
  {
    for ( int i = 0; i <= order; i++ ) {

      const int idxStart = idxEnd;

      if ( dim > 1 )
        idxEnd = _computeMonomialBasisGradientRecursion( order - i, x, res, idxEnd, dim - 1 );
      else {
        idxEnd++;
      }
      for ( int idx = idxStart; idx < idxEnd; idx++ ) {
        if ( i > 0 )
          res( idx, dim - 1 ) = i * integerPower( x[dim - 1], i - 1 );
        else
          res( idx, dim - 1 ) = 0;
      }
    }
    return idxEnd;
  }

  inline void computeMonomialBasis( int order, const Eigen::VectorXd& x, Eigen::VectorXd& res )
  {
    res.setOnes();
    _computeMonomialBasisRecursion( order, x, res, 0, x.size() );
  }

  inline void computeMonomialBasisGradient( int order, const Eigen::VectorXd& x, Eigen::MatrixXd& res )
  {
    res.setOnes();
    _computeMonomialBasisGradientRecursion( order, x, res, 0, x.size() );
  }

} // namespace Marmot::Math
