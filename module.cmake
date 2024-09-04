include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB sources_material "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
list(APPEND sources ${sources_material})


list(APPEND publicheaders 
    "${CMAKE_CURRENT_LIST_DIR}/include/Marmot/MarmotMeshfreeApproximation.h" 
    "${CMAKE_CURRENT_LIST_DIR}/include/Marmot/MarmotMeshfreeReproducingKernelApproximation.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/Marmot/MarmotMeshfreeKernelFunction.h" 
    "${CMAKE_CURRENT_LIST_DIR}/include/Marmot/MarmotMeshfreeKernelFunctionBSplineBoxed.h" 
    "${CMAKE_CURRENT_LIST_DIR}/include/Marmot/MarmotParticle.h" 
    "${CMAKE_CURRENT_LIST_DIR}/include/Marmot/MarmotParticleLibrary.h" 
    )
