using QCBase
using RDM
using FermiCG
using Printf
using Test
using JLD2


@testset "openshell_tpsci" begin
    @load "_testdata_cmf_h9.jld2"
    
    ref_fock = FockConfig(init_fspace)

    # Do TPS
    M=100
    cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], ref_fock, max_roots=M, verbose=1);
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=M, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

    nroots=3
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)

    ci_vector = FermiCG.add_spin_focksectors(ci_vector)

    display(ci_vector)
    eci, v = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);

    e0a, v0a = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
        thresh_cipsi = 1e-3, 
        thresh_foi   = 1e-5,
        thresh_asci  = -1);
    
    ept = FermiCG.compute_pt2_energy(v0a, cluster_ops, clustered_ham, thresh_foi=1e-8)
   
    tpsci_ref = [-14.05014150
                 -14.02155292
                 -14.00595447]
                 
    e2_ref = [-14.05028658
              -14.02164792
              -14.00602933]

    @test all(isapprox(tpsci_ref, e0a, atol=1e-8)) 
    @test all(isapprox(e2_ref, ept+e0a, atol=1e-8)) 
end

@testset "openshell_bst" begin
    @load "_testdata_cmf_h9.jld2"
    
    
    ref_fock = FockConfig(init_fspace)

    # Do TPS
    M=100
    cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], ref_fock, max_roots=M, verbose=1);
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=M, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

    nroots=3

    # BST
    #
    
    # start by defining P/Q spaces
    p_spaces = Vector{ClusterSubspace}()
   
    for ci in clusters
        ssi = ClusterSubspace(clusters[ci.idx])

        num_states_in_p_space = 1
        # our clusters are near triangles, with degenerate gs, so keep two states
        add_subspace!(ssi, ref_fock[ci.idx], 1:num_states_in_p_space)
        add_subspace!(ssi, (ref_fock[ci.idx][2], ref_fock[ci.idx][1]), 1:num_states_in_p_space) # add flipped spin
        push!(p_spaces, ssi)
    end

    ci_vector = BSTstate(clusters, p_spaces, cluster_bases, R=3) 
    
    na = 5
    nb = 4
    FermiCG.fill_p_space!(ci_vector, na, nb)
    FermiCG.eye!(ci_vector)
    e_ci, v = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham)

    ept = FermiCG.compute_pt2_energy(v, cluster_ops, clustered_ham, thresh_foi=1e-8)
    
    if true 
        e_var, v_var = block_sparse_tucker( v, cluster_ops, clustered_ham,
                                           max_iter    = 20,
                                           max_iter_pt = 200,
                                           nbody       = 4,
                                           H0          = "Hcmf",
                                           thresh_var  = 1e-1,
                                           thresh_foi  = 1e-6,
                                           thresh_pt   = 1e-3,
                                           ci_conv     = 1e-5,
                                           ci_max_iter = 100,
                                           do_pt       = true,
                                           resolve_ss  = false,
                                           tol_tucker  = 1e-4,
                                           solver      = "davidson")

        e_ref = [-14.050153133150385
                 -14.021579538798385
                 -14.00597811927653]
        @test all(isapprox(e_ref, e_var, atol=1e-8)) 
    end
end

