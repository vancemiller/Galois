/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/**
 * @file DistributedGraphLoader.cpp
 *
 * Contains definitions for command line arguments related to distributed
 * graph loading.
 */

#include <galois/graphs/DistributedGraphLoader.h>

using namespace galois::graphs;

namespace cll = llvm::cl;

cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"),
                                cll::Required);
cll::opt<std::string> inputFileTranspose("graphTranspose",
                                         cll::desc("<input file, transposed>"),
                                         cll::init(""));
cll::opt<bool>
    inputFileSymmetric("symmetricGraph",
                       cll::desc("Set this flag if graph is symmetric"),
                       cll::init(false));
cll::opt<std::string> partFolder("partFolder",
                                 cll::desc("path to partitionFolder "
                                           "(deprecated, do not use)"),
                                 cll::init(""), cll::Hidden);
cll::opt<PARTITIONING_SCHEME> partitionScheme(
    "partition", cll::desc("Type of partitioning."),
    cll::values(
        clEnumValN(OEC, "oec", "Outgoing Edge-Cut (default)"),
        clEnumValN(IEC, "iec", "Incoming Edge-Cut"),
        clEnumValN(HOVC, "hovc", "Outgoing Hybrid Vertex-Cut"),
        clEnumValN(HIVC, "hivc", "Incoming Hybrid Vertex-Cut"),
        clEnumValN(BOARD2D_VCUT, "2dvc", "2d Checkerboard Vertex-Cut"),
        clEnumValN(CART_VCUT, "cvc", "Cartesian Vertex-Cut"),
        clEnumValN(JAGGED_CYCLIC_VCUT, "jcvc", "Jagged Cyclic Vertex-Cut"),
        clEnumValN(JAGGED_BLOCKED_VCUT, "jbvc", "Jagged Blocked Vertex-Cut"),
        clEnumValN(OVER_DECOMPOSE_2_VCUT, "od2vc",
                   "Over decomposed by 2 cartesian Vertex-Cut"),
        clEnumValN(OVER_DECOMPOSE_4_VCUT, "od4vc",
                   "Over decomposed by 4 cartesian Vertex-Cut"),
        clEnumValN(CEC, "cec", "Custom edge cut from vertexID mapping"),
        clEnumValEnd),
    cll::init(OEC));
cll::opt<unsigned int>
    VCutThreshold("VCutThreshold",
                  cll::desc("Threshold for high degree edges."),
                  cll::init(1000), cll::Hidden);

cll::opt<std::string>
    vertexIDMapFileName("vertexIDMapFileName",
                        cll::desc("<file containing the "
                                  "vertexID to hosts mapping for "
                                  "the custom edge cut.>"),
                        cll::init(""), cll::Hidden);

cll::opt<bool> readFromFile("readFromFile",
                            cll::desc("Set this flag if graph is to be "
                                      "constructed from file (file must be "
                                      "created by Abelian CSR)"),
                            cll::init(false), cll::Hidden);

cll::opt<std::string>
    localGraphFileName("localGraphFileName",
                       cll::desc("Name of the local file to construct "
                                 "local graph (file must be created by "
                                 "Abelian CSR)"),
                       cll::init("local_graph"), cll::Hidden);

cll::opt<bool> saveLocalGraph("saveLocalGraph",
                              cll::desc("Set to save the local CSR graph"),
                              cll::init(false), cll::Hidden);
