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
include("Tensors.jl")
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
include("ClusteredStates.jl")

include("tucker_inner.jl")
include("tucker_outer.jl")

include("tpsci_functions.jl")
include("Tucker_functions.jl")
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
end
