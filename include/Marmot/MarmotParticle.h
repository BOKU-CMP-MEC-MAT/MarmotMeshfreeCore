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

#include "Marmot/MarmotMeshfreeKernelFunction.h"
#include "Marmot/MarmotUtils.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace Marmot::Meshfree {

  /**
   * @brief The MarmotParticle class is the base class for all particles in the meshfree framework.
   * It provides the interface for the interaction with the meshfree framework.
   *
   * The MarmotParticle class is the base class for all particles in the meshfree framework.
   * It interacts with degrees of freedom (living on the nodes) via shape functions (MarmotMeshfreeShapeFunction).
   *
   * Similar to finite elements, the MarmotParticle class is responsible for the computation of the internal forces,
   * the body loads and the distributed loads.
   * Also, it may have a distinct shape (commonly a point).
   */

  class MarmotParticle {

  public:
    MarmotParticle() = default;

    virtual ~MarmotParticle() = default;

    /* virtual void assignApproximationType( const MarmotMeshfreeApproximation& approximation ) = 0; */

    virtual void assignMeshfreeKernelFunctions(
      const std::vector< const MarmotMeshfreeKernelFunction* >& kernelFunctions ) = 0;

    virtual void assignStateVars( double* stateVars, int nStateVars ) = 0;

    /* virtual void assignMaterial( const MarmotMaterialSection& material ) = 0; */

    virtual void initializeYourself() = 0;

    virtual void acceptStateAndPosition() {};

    virtual StateView getStateView( const std::string& stateName ) const = 0;

    virtual int getNumberOfRequiredStateVars() const = 0;

    virtual int getDimension() const = 0;

    virtual int getNumberOfVertices() const = 0;

    /// Get the shape of the material point (e.g., point, line, triangle, tetrahedron) in Ensight Gold notation
    virtual std::string getParticleShape() const = 0;

    /// Get the coordinates of the vertices of the particle (e.g., the nodes of a tetrahedron) for visualization
    virtual void getVisualizationVertexCoordinates( double* coordinates ) const = 0;

    /// Get the coordinates of the vertices of the particle (e.g., the nodes of a tetrahedron) for testing the coverage
    /// of the particle by shape functions.
    virtual void getVertexCoordinates( double* coordinates ) const = 0;

    /// Get the fields on which the particle lives (e.g., displacement, temperature, ...). For a particle, the
    /// contribution to each attached node is equal.
    virtual const std::vector< std::string >& getFields() const = 0;

    /// Get the permutation pattern of the degrees of freedom (e.g., if a blocked storage is used)
    virtual const std::vector< int >& getDofIndicesPermutationPattern() const = 0;

    virtual void computePhysicsKernels( const double* dQ,
                                        double*       fInt,
                                        double*       dFInt_ddQ,
                                        double        timeNew,
                                        double        dT ) = 0;

    virtual void computeBodyLoad( int           type,
                                  const double* load,
                                  double*       fExt,
                                  double*       dExt_dQ,
                                  double        timeNew,
                                  double        dT ) const = 0;

    virtual void computeDistributedLoad( int           type,
                                         int           surfaceID,
                                         const double* load,
                                         const double* dQ,
                                         double*       fExt,
                                         double*       dExt_dQ,
                                         double        timeNew,
                                         double        dT ) const = 0;

    virtual void getInterpolationVector( double* vec, const double* coordinates ) const = 0;

    virtual const std::unordered_map< std::string, int >& getSupportedBodyLoadTypes() const = 0;

    virtual const std::unordered_map< std::string, int >& getSupportedDistributedLoadTypes() const = 0;
  };

} // namespace Marmot::Meshfree
