using FermiCG
using Test
using Random

Random.seed!(1234567)

@testset "FermiCG" begin
    include("test_hamiltonians.jl")
    include("test_cmf.jl")
    include("test_Clusters.jl")
    include("test_FCI.jl")
    include("test_davidson.jl")
    include("test_TDMs.jl")
    include("test_tpsci.jl")
end
