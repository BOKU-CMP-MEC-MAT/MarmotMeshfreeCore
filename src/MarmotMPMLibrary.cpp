#include "Marmot/MarmotMPMLibrary.h"
#include "Marmot/MarmotCell.h"
#include "Marmot/MarmotJournal.h"
#include "Marmot/MarmotMaterialPoint.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>

namespace MarmotLibrary {

  std::string makeStringUpperCase( const std::string& str )
  {
    std::string strUpperCase = str;

    // TODO: MOVE TO MarmotUtils or similar

    std::transform( strUpperCase.begin(), strUpperCase.end(), strUpperCase.begin(), toupper );

    return strUpperCase;
  }

  // MaterialPointFactory

  std::unordered_map< std::string, MarmotMaterialPointFactory::materialPointFactoryFunction >
    MarmotMaterialPointFactory::materialPointFactoryFunctionByName;

  bool MarmotMaterialPointFactory::registerMaterialPoint( const std::string&           materialPointName,
                                                          materialPointFactoryFunction factoryFunction )
  {

    const auto materialPointNameUpperCase = makeStringUpperCase( materialPointName );

    assert( materialPointFactoryFunctionByName.find( materialPointNameUpperCase ) ==
            materialPointFactoryFunctionByName.end() );

    materialPointFactoryFunctionByName[materialPointNameUpperCase] = factoryFunction;

    return true;
  }

  MarmotMaterialPoint* MarmotMaterialPointFactory::createMaterialPoint( const std::string& materialPointName,
                                                                        int                materialPointNumber,
                                                                        const double*      vertexCoordinates,
                                                                        int                sizeVertexCoordinates,
                                                                        double             volume
                                                                        /* const MarmotMaterialSection& material */
  )
  {
    const auto materialPointNameUpperCase = makeStringUpperCase( materialPointName );

    try {
      return materialPointFactoryFunctionByName.at(
        materialPointNameUpperCase )( materialPointNumber, vertexCoordinates, sizeVertexCoordinates, volume
                                      /* material */
      );
    }
    catch ( const std::out_of_range& e ) {
      throw std::invalid_argument( MakeString() << "Invalid materialPoint " << materialPointName << " requested!" );
    }
  }

  // ElementFactory

  std::unordered_map< std::string, MarmotCellFactory::cellFactoryFunction >
    MarmotCellFactory::cellFactoryFunctionByName;

  bool MarmotCellFactory::registerCell( const std::string& cellName, cellFactoryFunction factoryFunction )
  {

    const auto cellNameUpperCase = makeStringUpperCase( cellName );

    assert( cellFactoryFunctionByName.find( cellNameUpperCase ) == cellFactoryFunctionByName.end() );

    cellFactoryFunctionByName[cellNameUpperCase] = factoryFunction;

    return true;
  }

  MarmotCell* MarmotCellFactory::createCell( const std::string& cellName,
                                             int                cellNumber,
                                             const double*      nodeCoordinates,
                                             int                sizeNodeCoordinates )
  {
    const auto cellNameUpperCase = makeStringUpperCase( cellName );

    try {
      return cellFactoryFunctionByName.at( cellNameUpperCase )( cellNumber, nodeCoordinates, sizeNodeCoordinates );
    }
    catch ( const std::out_of_range& e ) {
      throw std::invalid_argument( MakeString() << "Invalid cell " << cellName << " requested!" );
    }
  }

} // namespace MarmotLibrary
