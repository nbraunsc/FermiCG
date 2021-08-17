using FermiCG
using Printf
using Test
using Random
using LinearAlgebra 

#@testset "Tuck" begin
    atoms = []
    clusters = []
    na = 0
    nb = 0
    init_fspace = []
    
    function generate_H_ring(n,radius)
        theta = 2*pi/n

        atoms = []
        for i in 0:n-1
            push!(atoms,Atom(i+1,"H",[radius*cos(theta*i), radius*sin(theta*i), 0]))
        end
        return atoms
    end

    #
    # Test basic Tucker stuff
    Random.seed!(2);
    A = rand(4,6,3,3,5)
    #tuck = FermiCG.Tucker(A, thresh=20, verbose=1)
    tuck = FermiCG.Tucker((A,), thresh=10, verbose=1)
    
    display(size.(tuck.core))
    display(size.(tuck.factors))
    B = FermiCG.recompose(tuck)
    println()
    println(FermiCG.dims_small(tuck))
    println(FermiCG.dims_large(tuck))
    @test all(FermiCG.dims_small(tuck) .== [1, 1, 1, 1, 1])
    @test all(FermiCG.dims_large(tuck) .== [4, 6, 3, 3, 5])
     
    A = rand(4,6,3,3,5)
    tuck = FermiCG.Tucker(A, thresh=-1, verbose=1)
    B = FermiCG.recompose(tuck)
    @test isapprox(abs.(A), abs.(B[1]), atol=1e-12)

    A = rand(4,6,3,3,5)*.1
    B = rand(4,6,3,3,5)*.1
    C = A+B

    #tuckA = FermiCG.Tucker(A, thresh=-1, verbose=1, max_number=2)
    #tuckB = FermiCG.Tucker(B, thresh=-1, verbose=1, max_number=2)
    #tuckC = FermiCG.Tucker(C, thresh=-1, verbose=1, max_number=2)
    tuckA = FermiCG.Tucker(A, thresh=-1, verbose=0)
    tuckB = FermiCG.Tucker(B, thresh=-1, verbose=0)
    tuckC = FermiCG.Tucker(C, thresh=-1, verbose=0)

    # test Tucker addition
    test = tuckA + tuckB
    @test isapprox(FermiCG.dot(tuckC,tuckC), FermiCG.dot(test,test), atol=1e-12)


    #
    # Now test basis transformation
    A = rand(4,6,3,3,5)*.1
    
    trans1 = Dict{Int,Matrix{Float64}}() 
    trans1[2] = rand(6,5)
    trans1[4] = rand(3,2)

    trans2 = Vector{Matrix{Float64}}([])
    for i = 1:5
        if haskey(trans1, i)
            push!(trans2, trans1[i])
        else
            push!(trans2, Matrix(1.0I,size(A,i),size(A,i)))
        end
    end

    display((length(A), size(A)))
    
    A1 = FermiCG.transform_basis(A, trans1)
    display((length(A1), size(A1)))
    
    A2 = FermiCG.transform_basis(A, trans2)
    display((length(A1), size(A2)))
    
    #A2 = FermiCG.tucker_recompose(A, trans2)
    #display((length(A2), size(A2)))
    
    
#end

