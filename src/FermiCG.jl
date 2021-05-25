"""
General electronic course-graining platform
"""
module FermiCG


#####################################
# External packages
#
using Compat
using HDF5
using KrylovKit
using LinearAlgebra
using NDTensors
using PackageCompiler
using Parameters
using Printf
using TimerOutputs
using BenchmarkTools 
using OrderedCollections 
using IterTools
# using Unicode
#
#####################################



#####################################
# Local Imports
#
include("Utils.jl")
include("hosvd.jl")
include("StringCI/StringCI.jl");
include("Solvers.jl");
include("Hamiltonians.jl");
include("Clusters.jl")
include("Indexing.jl")
include("States.jl")

include("FockSparse_ElementSparse.jl")
include("FockSparse_BlockSparse.jl")
include("FockSparse_BlockSparseTucker.jl")

include("ClusteredTerms.jl")
#include("ClusteredStates.jl")

include("tucker_inner.jl")
include("tucker_outer.jl")
include("bst.jl")

include("tpsci_inner.jl")
include("tpsci_matvec_thread.jl")
include("tpsci_outer.jl")

include("dense_inner.jl")
include("dense_outer.jl")

include("CompressedTucker.jl")
include("CMFs.jl")
include("pyscf/PyscfFunctions.jl");
#
#####################################

export StringCI
export InCoreInts
export Molecule
export Atom
export Cluster
export ClusterBasis
export ClusteredState
export ClusterConfig 
export FockConfig 

end
