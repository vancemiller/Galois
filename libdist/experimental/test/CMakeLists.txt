#makeTest(pairingheap)
#makeTest(dist)
makeTest(ADD_TARGET distforeach DISTSAFE COMMAND_PREFIX ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2)
makeTest(ADD_TARGET thirdgraph DISTSAFE COMMAND_PREFIX ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2)
#makeTest(ADD_TARGET foreachtask)
makeTest(ADD_TARGET networktest DISTSAFE COMMAND_PREFIX ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2)
#makeTest(ADD_TARGET reduce)
makeTest(ADD_TARGET bag DISTSAFE)
makeTest(ADD_TARGET pairingheap DISTSAFE)
makeTest(ADD_TARGET skiplist DISTSAFE)
#makeTest(ADD_TARGET networkbe DISTSAFE)
makeTest(ADD_TARGET networkfe DISTSAFE)
makeTest(ADD_TARGET netbandwidth DISTSAFE COMMAND_PREFIX ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2)
#makeTest(ADD_TARGET oneach DISTSAFE COMMAND_PREFIX ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2)
makeTest(ADD_TARGET serialize DISTSAFE)
makeTest(ADD_TARGET serialize-speed DISTSAFE)
makeTest(ADD_TARGET fatpointerc DISTSAFE)
#makeTest(ADD_TARGET gasnet)
#makeTest(ADD_TARGET dirtest)
makeTest(ADD_TARGET CacheManagerTest DISTSAFE)
makeTest(ADD_TARGET LCDist DISTSAFE)
#makeTest(ADD_TARGET ping DISTSAFE)
makeTest(ADD_TARGET netraw DISTSAFE)
makeTest(ADD_TARGET netbuffered DISTSAFE)
makeTest(ADD_TARGET mpi DISTSAFE)
makeTest(ADD_TARGET netband2 DISTSAFE)
makeTest(ADD_TARGET netbandmpi DISTSAFE)

#app(TestNetBandWidth netbandwidth.cpp DISTSAFE)
makeTest(ADD_TARGET vertexCut DISTSAFE)
makeTest(ADD_TARGET serialize-vectors DISTSAFE)
