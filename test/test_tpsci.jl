using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using HDF5
using Random
using PyCall
using Arpack

@testset "tpsci" begin
    atoms = []

    r = 1
    a = 1 
    push!(atoms,Atom(1,"H", [0, 0*a, 0*r]))
    push!(atoms,Atom(2,"H", [0, 0*a, 1*r]))
    push!(atoms,Atom(3,"H", [0, 1*a, 2*r]))
    push!(atoms,Atom(4,"H", [0, 1*a, 3*r]))
    push!(atoms,Atom(5,"H", [0, 2*a, 4*r]))
    push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
    push!(atoms,Atom(7,"H", [0, 3*a, 6*r]))
    push!(atoms,Atom(8,"H", [0, 3*a, 7*r]))
    push!(atoms,Atom(9,"H", [0, 4*a, 8*r]))
    push!(atoms,Atom(10,"H",[0, 4*a, 9*r]))
    push!(atoms,Atom(11,"H",[0, 5*a, 10*r]))
    push!(atoms,Atom(12,"H",[0, 5*a, 11*r]))


    clusters    = [(1:2),(3:4),(5:6),(7:8),(9:12)]
    init_fspace = [(1,1),(1,1),(1,1),(1,1),(2,2)]
    clusters    = [(1:4),(5:8),(9:12)]
    init_fspace = [(2,2),(2,2),(2,2)]
    na = 6
    nb = 6


    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)

    nroots = 1

    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
	
    @printf(" Do FCI\n")
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 200 
    cisolver.conv_tol = 1e-8
    nelec = na + nb
    norb = size(ints.h1,1)
    #e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots)

    #e_fci = [-18.33022092,
    #         -18.05457644]
    e_fci  = [-18.33022092,
              -18.05457645,
              -18.02913047,
              -17.99661027
             ]

    for i in 1:length(e_fci)
        @printf(" %4i %12.8f %12.8f\n", i, e_fci[i], e_fci[i]+ints.h0)
    end


    # localize orbitals
    C = mf.mo_coeff
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    #FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    #
    # define clusters
    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))

    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, 
                                       max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    #FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    ints = FermiCG.orbital_rotation(ints,U)

    e_ref = e_cmf - ints.h0

    max_roots = 100
    # build Hamiltonian, cluster_basis and cluster ops
    #display(Da)
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots, 
                                                       init_fspace=init_fspace, rdm1a=Da, rdm1b=Db)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);



    p_spaces = Vector{FermiCG.ClusterSubspace}()
    q_spaces = Vector{FermiCG.ClusterSubspace}()

    # define p spaces
    for ci in clusters
        tss = FermiCG.ClusterSubspace(ci)
        tss[init_fspace[ci.idx]] = 1:1
        push!(p_spaces, tss)
    end

    # define q spaces
    for tssp in p_spaces 
        tss = FermiCG.get_ortho_compliment(tssp, cluster_bases[tssp.cluster.idx])
        push!(q_spaces, tss)
    end

    println(" ================= Cluster P Spaces ===================")
    display.(p_spaces)
    println(" ================= Cluster Q Spaces ===================")
    display.(q_spaces)

    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);

    if true 

        ci_vector = FermiCG.ClusteredState(clusters, R=nroots)

        ref_fock = FermiCG.FockConfig(init_fspace)
        FermiCG.add_fockconfig!(ci_vector, ref_fock)

        @time e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, 
                                  thresh_cipsi=1e-3, thresh_foi=1e-9, thresh_asci=1e-4, conv_thresh=1e-5, matvec=1);

        ref = [-18.32973618]

        @test isapprox(abs.(ref), abs.(e0), atol=1e-8)
    end
   
    if true 
        nroots = 4

        ci_vector = FermiCG.ClusteredState(clusters, R=nroots)

        ref_fock = FermiCG.FockConfig(init_fspace)
        FermiCG.add_fockconfig!(ci_vector, ref_fock)

        #1 excitons 
        ci_vector[ref_fock][ClusterConfig([2,1,1])] = [0,1,0,0]
        ci_vector[ref_fock][ClusterConfig([1,2,1])] = [0,0,1,0]
        ci_vector[ref_fock][ClusterConfig([1,1,2])] = [0,0,0,1]

        e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false,
                                  thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);

        e2, v1 = FermiCG.compute_pt2(v0, cluster_ops, clustered_ham, thresh_foi=1e-8, matvec=3)

        ref = [-18.32932467
               -18.05349474
               -18.02775313
               -17.99514933
              ]
        @test isapprox(abs.(ref), abs.(e0+e2), atol=1e-7)


        rotations = FermiCG.hosvd(v0, cluster_ops)
        for ci in clusters
            FermiCG.rotate!(cluster_ops[ci.idx], rotations[ci.idx])
            FermiCG.rotate!(cluster_bases[ci.idx], rotations[ci.idx])
            FermiCG.check_basis_orthogonality(cluster_bases[ci.idx])
        end

        #cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
        #FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);


        e0a, v0a = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false, 
                                    thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2);
        e2a, v1a = FermiCG.compute_pt2(v0a, cluster_ops, clustered_ham, thresh_foi=1e-8, matvec=3)

        ref = [-18.32916288
               -18.05357935
               -18.02800015
               -17.99499973]

        @test isapprox(abs.(ref), abs.(e0a+e2a), atol=1e-7)
    end

    ci_vector = FermiCG.ClusteredState(clusters, R=4)
    ref_fock = FermiCG.FockConfig(init_fspace)
    FermiCG.add_fockconfig!(ci_vector, ref_fock)
    ci_vector[ref_fock][ClusterConfig([2,1,1])] = [0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1])] = [0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2])] = [0,0,0,1]

    sig1 = FermiCG.open_matvec_serial2(ci_vector, cluster_ops, clustered_ham, nbody=4, thresh=1e-8)
    sig2 = FermiCG.open_matvec_thread(ci_vector, cluster_ops, clustered_ham, nbody=4, thresh=1e-8)
    sig3 = FermiCG.open_matvec_thread2(ci_vector, cluster_ops, clustered_ham, nbody=4, thresh=1e-8)
        
    @test isapprox(norm(sig1), norm(sig2), atol=1e-16)
    @test isapprox(norm(sig1), norm(sig3), atol=1e-16)
end

