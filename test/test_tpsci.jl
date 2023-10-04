using QCBase
using RDM
using FermiCG
using Printf
using Test
using LinearAlgebra
using Random
using Arpack
using JLD2

@testset "tpsci he 64bit" begin
    @load "_testdata_cmf_he4.jld2"
    
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
    
    nroots = 5

    ref_fock = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)

    #1 excitons 
    ci_vector[ref_fock][ClusterConfig([2,1,1,1])] = [0,1,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1,1])] = [0,0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2,1])] = [0,0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,1,2])] = [0,0,0,0,1]

    #e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false,
    #                          thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, ci_conv=1e-10,
                              thresh_cipsi=1e-3, thresh_foi=1e-8, thresh_asci=-1, conv_thresh=1e-7, ci_lindep_thresh=1e-12);
    
    if true
        H = FermiCG.build_full_H(v0, cluster_ops, clustered_ham)
        sig1 = H*FermiCG.get_vector(v0)
        sig2 = FermiCG.tps_ci_matvec(v0, cluster_ops, clustered_ham)

        @test isapprox(norm(sig1-sig2), 0.0, atol=1e-12) 

        guess = deepcopy(v0)
        FermiCG.randomize!(guess)
        FermiCG.orthonormalize!(guess)
        e0b, v0b = FermiCG.tps_ci_direct(guess, cluster_ops, clustered_ham, conv_thresh=1e-10);
        e0c, v0c = FermiCG.tps_ci_davidson(guess, cluster_ops, clustered_ham, conv_thresh=1e-9, precond=false, max_iter=200, lindep_thresh=1e-14);
        e0d, v0d = FermiCG.tps_ci_davidson(guess, cluster_ops, clustered_ham, conv_thresh=1e-9, precond=true, max_iter=200, lindep_thresh=1e-14);

        @test isapprox(abs.(e0), abs.(e0b), atol=1e-9)
        @test isapprox(abs.(e0), abs.(e0c), atol=1e-8)
        @test isapprox(abs.(e0), abs.(e0d), atol=1e-8)

    end
    e2 = FermiCG.compute_pt2_energy(v0, cluster_ops, clustered_ham, thresh_foi=1e-10)
    
    display(e0)
    display(e2)
    display(e0+e2)

    ref = [
        -16.886058279836007
        -15.435798355654288
        -15.4228104425694
        -15.422680127510118
        -15.409357434493316
        ]
    @test isapprox(abs.(ref), abs.(e0), atol=1e-6)
    
    ref = [
        -16.886190525742013
        -15.436190685557376
        -15.42326855339792
        -15.42302659988098
        -15.409737495164334
        ]
    @test isapprox(abs.(ref), abs.(e0+e2), atol=1e-6)

    e2a, v1a = FermiCG.compute_pt1_wavefunction(v0, cluster_ops, clustered_ham, thresh_foi=1e-8)
    @test isapprox(abs.(e2), abs.(e2a), atol=1e-7)
    

end
@testset "tpsci h12 64bit" begin
    @load "_testdata_cmf_h12_64bit.jld2"
    
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
    
    nroots = 7

    ref_fock = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)

    #1 excitons 
    ci_vector[ref_fock][ClusterConfig([2,1,1,1,1])] = [0,1,0,0,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1,1,1])] = [0,0,1,0,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2,1,1])] = [0,0,0,1,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,3,1,1])] = [0,0,0,0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,1,2,1])] = [0,0,0,0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,1,1,2])] = [0,0,0,0,0,0,1]

    #e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false,
    #                          thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, ci_conv=1e-8,
                              thresh_cipsi=1e-2, thresh_foi=1e-5, thresh_asci=-1, conv_thresh=1e-7);
    
    e2 = FermiCG.compute_pt2_energy(v0, cluster_ops, clustered_ham, thresh_foi=1e-10)
    
    display(e0)
    display(e2)
    display(e0+e2)

    ref = [
        -18.325122258888328
        -18.042608248491383
        -18.0162458168032
        -17.98626007894176
        -17.953886773652684
        -17.92637650450847
        -17.90934751388588
        ]
    @test isapprox(abs.(ref), abs.(e0), atol=1e-8)

    ref = [
        -18.329245064283377
        -18.052309500798415
        -18.026878527967494
        -17.994780942798855
        -17.962157723302557
        -17.934865155401628
        -17.91770685344104
        ]
    @test isapprox(abs.(ref), abs.(e0+e2), atol=1e-8)


end
@testset "tpsci h12 32bit" begin
    @load "_testdata_cmf_h12_32bit.jld2"
    
    max_roots = 20
    
    # Convert to 32bit
    ints = InCoreInts(ints, Float32)

    #
    # form Cluster data
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1, 
                                                       max_roots=max_roots, 
                                                       init_fspace=init_fspace, 
                                                       rdm1a=d1.a, rdm1b=d1.b, T=Float32)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);


    nroots = 4

    ref_fock = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float32)

    #1 excitons 
    ci_vector[ref_fock][ClusterConfig([2,1,1,1,1])] = [0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1,1,1])] = [0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2,1,1])] = [0,0,0,1]

    #e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false,
    #                          thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
                              thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    
    e2 = FermiCG.compute_pt2_energy(v0, cluster_ops, clustered_ham, thresh_foi=1e-8)

    display(e0)
    display(e2)
    display(e0+e2)
    ref = [
           -18.32923698
           -18.05237389
           -18.02698708
           -17.99495125
          ]
    @test isapprox(abs.(ref), abs.(e0+e2), atol=1e-4)
end
