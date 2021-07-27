using Distributed
using ThreadPools
using JLD2
using LinearMaps
#using IterativeSolvers

"""
    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    zero_fock = TransferConfig([(0,0) for i in ci_vector.clusters])
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            ket_idx = 0
            for (fock_ket, configs_ket) in ci_vector.data
                fock_trans = fock_bra - fock_ket

                # check if transition is connected by H
                if haskey(clustered_ham, fock_trans) == false
                    ket_idx += length(configs_ket)
                    continue
                end

                for (config_ket, coeff_ket) in configs_ket
                    ket_idx += 1
                    ket_idx <= bra_idx || continue


                    for term in clustered_ham[fock_trans]
                    
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                        
                        me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                        H[bra_idx, ket_idx] += me 
                    end

                    H[ket_idx, bra_idx] = H[bra_idx, ket_idx]

                end
            end
        end
    end
    return H
end
#=}}}=#


"""
    build_full_H_parallel(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H_parallel(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    jobs = []

    zero_fock = TransferConfig([(0,0) for i in ci_vector.clusters])
    bra_idx = 0
    N = length(ci_vector.clusters)
    

    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            #push!(jobs, (bra_idx, fock_bra, config_bra) )
            #push!(jobs, (bra_idx, fock_bra, config_bra, H[bra_idx,:]) )
            push!(jobs, (bra_idx, fock_bra, config_bra, zeros(dim)) )
        end
    end

    function do_job(job)
        fock_bra = job[2]
        config_bra = job[3]
        Hrow = job[4]
        ket_idx = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            if haskey(clustered_ham, fock_trans) == false
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, coeff_ket) in configs_ket
                ket_idx += 1
                ket_idx <= job[1] || continue

                for term in clustered_ham[fock_trans]
                       
                    #length(term.clusters) <= 2 || continue
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    
                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    Hrow[ket_idx] += me 
                    #H[job[1],ket_idx] += me 
                end

            end

        end
    end

    # because @threads divides evenly the loop, let's distribute thework more fairly
    #mid = length(jobs) ÷ 2
    #r = collect(1:length(jobs))
    #perm = [r[1:mid] reverse(r[mid+1:end])]'[:]
    #jobs = jobs[perm]
    
    #for job in jobs
    Threads.@threads for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    for job in jobs
        H[job[1],:] .= job[4]
    end

    for i in 1:dim
        @simd for j in i+1:dim
            @inbounds H[i,j] = H[j,i]
        end
    end


    return H
end
#=}}}=#


"""
    tps_ci_direct(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}

# Solve for eigenvectors/values in the basis defined by `ci_vector`. Use direct diagonalization. 
"""
function tps_ci_direct(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                      verbose   = 0) where {T,N,R}
    #={{{=#
    println()
    @printf(" |== Tensor Product State CI =======================================\n")
    vec_out = deepcopy(ci_vector)
    e0 = zeros(T,R)
    @printf(" Hamiltonian matrix dimension = %5i: \n", length(ci_vector))
    @printf(" %-50s", "Build full Hamiltonian matrix with dimension: ")
    @time H = build_full_H_parallel(vec_out, cluster_ops, clustered_ham)
    flush(stdout)
    @printf(" %-50s", "Diagonalize: ")
    if length(vec_out) > 1000
        @time e0,v = Arpack.eigs(H, nev = R, which=:SR)
    else
        @time F = eigen(H)
        e0 = F.values[1:R]
        v = F.vectors[:,1:R]
    end
    set_vector!(vec_out, v)

    clustered_S2 = extract_S2(ci_vector.clusters)
    @printf(" %-50s", "Compute S2 expectation values: ")
    @time s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e0[r], abs(s2[r]))
    end

    if verbose > 1
        for r in 1:R
            display(vec_out, root=r)
        end
    end

    @printf(" ==================================================================|\n")
    return e0, vec_out 
end
#=}}}=#



"""
    tps_ci_davidson(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}

# Solve for eigenvectors/values in the basis defined by `ci_vector`. Use iterative davidson solver. 
"""
function tps_ci_davidson(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                        v0 = nothing,
                        conv_thresh = 1e-5,
                        max_ss_vecs = 12,
                        max_iter    = 40,
                        shift       = nothing,
                        precond     = false,
                        verbose     = 0) where {T,N,R}
    #={{{=#
    println()
    @printf(" |== Tensor Product State CI =======================================\n")
    vec_out = deepcopy(ci_vector)
    e0 = zeros(T,R) 
   
    dim = length(ci_vector)
    iters = 0

    
    function matvec(v::AbstractMatrix)
        iters += 1
        #in = deepcopy(ci_vector) 
        in = ClusteredState(ci_vector, R=size(v,2))
        set_vector!(in, v)
        sig = deepcopy(in)
        zero!(sig)
        #build_sigma!(sig, ci_vector, cluster_ops, clustered_ham, cache=cache)
        return tps_ci_matvec(in, cluster_ops, clustered_ham)
    end

    Hmap = FermiCG.LinOp(matvec, dim)

    davidson = FermiCG.Davidson(Hmap, v0=get_vectors(ci_vector), 
                                max_iter=max_iter, max_ss_vecs=max_ss_vecs, nroots=R, tol=conv_thresh)

    e = nothing
    v = nothing
    if precond
        @printf(" %-50s", "Compute diagonal: ")
        clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = "Hcmf") 
        @time Hd = compute_diagonal(ci_vector, cluster_ops, clustered_ham_0)
        @printf(" %-50s", "Compute <0|H0|0>: ")
        @time E0 = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham_0)[1]
        @time Eref = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham)[1]
        Hd .+= Eref - E0
        @printf(" Now iterate: \n")
        flush(stdout)
        @time e,v = FermiCG.solve(davidson, Adiag=Hd);
    else
        @time e,v = FermiCG.solve(davidson);
    end
    set_vector!(vec_out, v)
    
    clustered_S2 = extract_S2(ci_vector.clusters)
    @printf(" %-50s", "Compute S2 expectation values: ")
    @time s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e[r], abs(s2[r]))
    end

    if verbose > 1
        for r in 1:R
            display(vec_out, root=r)
        end
    end

    @printf(" ==================================================================|\n")
    return e, vec_out 
end
#=}}}=#


"""
    tps_ci_matvec(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}

# Compute the action of `clustered_ham` on `ci_vector`. 
"""
function tps_ci_matvec(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    #={{{=#

    jobs = []

    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            push!(jobs, (bra_idx, fock_bra, config_bra, coeff_bra, zeros(T,R)) )
        end
    end

    function do_job(job)
        fock_bra = job[2]
        config_bra = job[3]
        coeff_bra = job[4]
        sig_out = job[5]
    
        for (fock_trans, terms) in clustered_ham
            fock_ket = fock_bra - fock_trans

            haskey(ci_vector.data, fock_ket) || continue
            
            configs_ket = ci_vector[fock_ket]


            for (config_ket, coeff_ket) in configs_ket
                for term in clustered_ham[fock_trans]
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
    
                    #norm(term.ints)*maximum(abs.(coeff_ket)) > 1e-5 || continue
                    #@btime norm($term.ints)*maximum(abs.($coeff_ket)) > 1e-12 
                    

                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    @simd for r in 1:R
                        @inbounds sig_out[r] += me * coeff_ket[r]
                    end
                    #@btime $sig_out .+= $me .* $ci_vector[$fock_ket][$config_ket] 
                end

            end

        end
    end

    #for job in jobs
    Threads.@threads for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    sigv = zeros(size(ci_vector))
    for job in jobs
        #for r in 1:R
        #    sigv[job[1],r] += job[5][r]
        #end
        sigv[job[1],:] .+= job[5]
    end

    return sigv
end
#=}}}=#



"""
    tpsci_ci(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
            thresh_cipsi = 1e-2,
            thresh_foi   = 1e-6,
            thresh_asci  = 1e-2,
            thresh_var   = -1.0,
            max_iter     = 10,
            conv_thresh  = 1e-4,
            nbody        = 4,
            incremental  = true,
            ci_conv      = 1e-5,
            ci_max_iter  = 50,
            ci_max_ss_vecs = 12,
            davidson     = false,
            max_mem_ci   = 20.0, 
            matvec       = 1) where {T,N,R}

# Run TPSCI 
- `thresh_cipsi`: threshold for which configurations to include in the variational space. Add if |c^{(1)}| > `thresh_cipsi`
- `thresh_foi`  : threshold for which terms to keep in the H|0> vector used to form the first order wavefunction
- `thresh_asci` : threshold for determining from which variational configurations  ``|c^{(0)}_i|`` > `thresh_asci` 
- `thresh_var`  : threshold for clipping the result of the variational wavefunction. Not really needed default set to -1 (off)
- `max_iter`    : maximum selected CI iterations
- `conv_thresh` : stop selected CI iterations when energy change is smaller than `conv_thresh`
- `nbody`       : only consider up to `nbody` terms when searching for new configurations
- `incremental` : for the sigma vector incrementally between iterations
- `matvec`      : which implementation of the matrix vector code
- `ci_conv`     : convergence threshold for the inner CI step (only needed when davidson is used)
- `ci_max_iter` : max iterations for inner CI step (only needed when davidson is used) 
- `ci_max_ss_vecs`: max subspace size for inner CI step (only needed when davidson is used) 
- `davidson`    : use davidson? changes to true after needing more than max_mem_ci
- `max_mem_ci`  : maximum memory (Gb) allowed for storing full H. If more is needed, do Davidson. 
"""
function tpsci_ci(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
    thresh_cipsi    = 1e-2,
    thresh_foi      = 1e-6,
    thresh_asci     = 1e-2,
    thresh_var      = -1.0,
    max_iter        = 10,
    conv_thresh     = 1e-4,
    nbody           = 4,
    incremental     = true,
    ci_conv         = 1e-5,
    ci_max_iter     = 50,
    ci_max_ss_vecs  = 12,
    davidson        = false,
    max_mem_ci      = 20.0, 
    matvec          = 1) where {T,N,R}
#={{{=#
    vec_var = deepcopy(ci_vector)
    vec_pt = deepcopy(ci_vector)
    length(ci_vector) > 0 || error(" input vector has zero length")
    zero!(vec_pt)
    e0 = zeros(T,R) 
    e2 = zeros(T,R) 
    e0_last = zeros(T,R)
    
    clustered_S2 = extract_S2(ci_vector.clusters)

    println(" ci_vector     : ", size(ci_vector) ) 
    println(" thresh_cipsi  : ", thresh_cipsi   ) 
    println(" thresh_foi    : ", thresh_foi     ) 
    println(" thresh_asci   : ", thresh_asci    ) 
    println(" thresh_var    : ", thresh_var     ) 
    println(" max_iter      : ", max_iter       ) 
    println(" conv_thresh   : ", conv_thresh    ) 
    println(" nbody         : ", nbody          ) 
    println(" incremental   : ", incremental    ) 
    println(" ci_conv       : ", ci_conv        ) 
    println(" ci_max_iter   : ", ci_max_iter    ) 
    println(" ci_max_ss_vecs: ", ci_max_ss_vecs ) 
    println(" davidson      : ", davidson       ) 
    println(" max_mem_ci    : ", max_mem_ci     ) 
    println(" matvec        : ", matvec         ) 
    
    vec_asci_old = ClusteredState(ci_vector.clusters, R=R)
    sig = ClusteredState(ci_vector.clusters, R=R)
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = "Hcmf") 

    for it in 1:max_iter

        println()
        println()
        println(" ===================================================================")
        @printf("     Selected CI Iteration: %4i epsilon: %12.8f\n", it,thresh_cipsi)
        println(" ===================================================================")

        if it > 1
            l1 = length(vec_var)
            clip!(vec_var, thresh=thresh_var)
            l2 = length(vec_var)
            @printf(" Clip values < %8.1e         %6i → %6i\n", thresh_var, l1, l2)
            
            l1 = length(vec_var)
            zero!(vec_pt)
            add!(vec_var, vec_pt)
            l2 = length(vec_var)
            @printf(" Add pt vector to current space %6i → %6i\n", l1, l2)
        else
            rand!(vec_var)
        end

        e0 = nothing
        mem_needed = sizeof(T)*length(vec_var)*length(vec_var)*1e-9
        @printf(" Memory needed to hold full CI matrix: %12.8f (Gb)\n",mem_needed)
        flush(stdout)
        if (mem_needed > max_mem_ci) || davidson == true
            orthonormalize!(vec_var)
            e0, vec_var = tps_ci_davidson(vec_var, cluster_ops, clustered_ham,
                                   conv_thresh = ci_conv,
                                   max_iter = ci_max_iter,
                                   max_ss_vecs = ci_max_ss_vecs)
        else
            e0, vec_var = tps_ci_direct(vec_var, cluster_ops, clustered_ham)
        end
        flush(stdout)
      


        # get barycentric energy <0|H0|0>
        Efock = compute_expectation_value_parallel(vec_var, cluster_ops, clustered_ham_0)
        #Efock = nothing
        flush(stdout)
        vec_asci = deepcopy(vec_var)
        l1 = length(vec_asci)
        clip!(vec_asci, thresh=thresh_asci)
        l2 = length(vec_asci)
        @printf(" Length of ASCI vector %8i → %8i \n", l1, l2)

        #
        # |sig_i> = H|v_i> = H|v_i-1> + H ( |v_i> - |v_i-1> )
        #                  = |sig_i-1> + H|del_i>

        del_v0 = deepcopy(vec_asci_old)
        ovlps = []
        for r in 1:R
            if dot(vec_asci_old, vec_asci, r, r) < 0
                scale!(vec_asci, -1.0, root=r)
            end
        end
        scale!(del_v0,-1.0)
        add!(del_v0, vec_asci)
        println(" Norm of delta v:")
        [@printf(" %12.8f\n",i) for i in norm(del_v0)]

        vec_asci_old = deepcopy(vec_asci)

        #_, vec_pt_del = compute_pt2(del_v0, cluster_ops, clustered_ham, E0=Efock, thresh_foi=thresh_foi, matvec=matvec, nbody=nbody)
        #add!(vec_pt, vec_pt_del)
        if incremental 
            if matvec == 1
                del_sig_it = open_matvec_serial2(del_v0, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
            elseif matvec == 2
                del_sig_it = open_matvec_thread(del_v0, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
            elseif matvec == 3
                del_sig_it = open_matvec_thread2(del_v0, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
            else
                error("wrong matvec")
            end
            flush(stdout)
            add!(sig, del_sig_it)

            project_out!(sig, vec_asci)
            println(" Length of FOIS vector: ", length(sig))


            @printf(" %-50s", "Compute diagonal: ")
            flush(stdout)
            @time Hd = compute_diagonal(sig, cluster_ops, clustered_ham_0)

            sig_v = get_vectors(sig)
            v_pt  = zeros(size(sig_v))

            println()
            @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)") 
            for r in 1:R
                denom = 1.0 ./ (Efock[r] .- Hd)  
                v_pt[:,r] .= denom .* sig_v[:,r] 
                e2[r] = sum(sig_v[:,r] .* v_pt[:,r])

                @printf(" %5s %12.8f %12.8f\n",r, e0[r], e0[r] + e2[r])
            end
            flush(stdout)

            vec_pt = deepcopy(sig)
            set_vector!(vec_pt,v_pt)
        else
            e2, vec_pt = compute_pt2(vec_asci, cluster_ops, clustered_ham, E0=Efock, thresh_foi=thresh_foi, matvec=matvec, nbody=nbody)
        end
        
        l1 = length(vec_pt)
        clip!(vec_pt, thresh=thresh_cipsi)
        l2 = length(vec_pt)
        @printf(" Length of PT1  vector %8i → %8i \n", l1, l2)
        #add!(vec_var, vec_pt)

        if maximum(abs.(e0_last .- e0)) < conv_thresh
            print_tpsci_iter(vec_var, it, e0, true)
            break
        else
            print_tpsci_iter(vec_var, it, e0, false)
            e0_last .= e0
        end
    end
    return e0, vec_var 
end
#=}}}=#


function print_tpsci_iter(ci_vector::ClusteredState{T,N,R}, it, e0, converged) where {T,N,R}
#={{{=#
    if converged 
        @printf("*TPSCI Iter %-3i Dim: %-6i", it, length(ci_vector))
    else
        @printf(" TPSCI Iter %-3i Dim: %-6i", it, length(ci_vector))
    end
    @printf(" E(var): ")
    for i in 1:R
        @printf("%13.8f ", e0[i])
    end
#    @printf(" E(pt2): ")
#    for i in 1:R
#        @printf("%13.8f ", e2[i])
#    end
    println()
end
#=}}}=#

"""
    compute_pt2(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=false,
        verbose=1,
        matvec=3) where {T,N,R}
"""
function compute_pt2(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=false,
        verbose=1,
        matvec=3) where {T,N,R}
    #={{{=#

    println()
    println(" |............................do PT2................................")
    println(" thresh_foi    :", thresh_foi   ) 
    println(" prescreen     :", prescreen   ) 
    println(" H0            :", H0   ) 
    println(" nbody         :", nbody   ) 

    e2 = zeros(T,R)
    
    norms = norm(ci_vector);
    println(" Norms of input states")
    [@printf(" %12.8f\n",i) for i in norms]
    println(" Compute FOIS vector")

    if matvec == 1
        #@time sig = open_matvec(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
        sig = open_matvec_serial2(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    elseif matvec == 2
        sig = open_matvec_thread(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    elseif matvec == 3
        sig = open_matvec_thread2(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    else
        error("wrong matvec")
    end
    #@time sig = open_matvec_parallel(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    #@btime sig = open_matvec_parallel($ci_vector, $cluster_ops, $clustered_ham, nbody=$nbody, thresh=$thresh_foi)
    println(" Length of FOIS vector: ", length(sig))

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
    
    project_out!(sig, ci_vector)
    println(" Length of FOIS vector: ", length(sig))
    
    @printf(" %-50s", "Compute diagonal")
    @time Hd = compute_diagonal(sig, cluster_ops, clustered_ham_0)
    
    if E0 == nothing
        @printf(" %-50s", "Compute <0|H0|0>:")
        @time E0 = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham_0)
        #E0 = diag(E0)
        flush(stdout)
    end

    @printf(" %-50s", "Compute <0|H|0>:")
    @time Evar = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham)
    #Evar = diag(Evar)
    flush(stdout)
    

    sig_v = get_vectors(sig)
    v_pt  = zeros(size(sig_v))

    println()
    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)") 
    for r in 1:R
        denom = 1.0 ./ (E0[r]/(norms[r]*norms[r]) .- Hd)  
        v_pt[:,r] .= denom .* sig_v[:,r] 
        e2[r] = sum(sig_v[:,r] .* v_pt[:,r])
   
        @printf(" %5s %12.8f %12.8f\n",r, Evar[r]/norms[r], Evar[r]/(norms[r]*norms[r]) + e2[r])
    end

    set_vector!(sig,v_pt)
    println(" ..................................................................|")

    return e2, sig 
end
#=}}}=#

"""
    compute_expectation_value(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; nbody=4) where {T,N,R}

Compute expectation value of a `ClusteredOperator` (`clustered_ham`) for state `ci_vector`
"""
function compute_expectation_value(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; nbody=4) where {T,N,R}
    #={{{=#

    out = zeros(T,R)

    for (fock_bra, configs_bra) in ci_vector.data

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket

                    me = 0.0
                    for term in clustered_ham[fock_trans]

                        length(term.clusters) <= nbody || continue
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                        me += contract_matrix_element(term, cluster_ops, 
                                                      fock_bra, config_bra, 
                                                      fock_ket, config_ket)
                    end

                    #out .+= coeff_bra .* coeff_ket .* me
                    for r in 1:R
                        out[r] += coeff_bra[r] * coeff_ket[r] * me
                    end

                end

            end
        end
    end

    return out 
end
#=}}}=#

"""
    function compute_expectation_value_parallel(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
"""
function compute_expectation_value_parallel(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    #={{{=#

    # 
    # This will be were we collect our results
    evals = zeros(T,R)

    jobs = []

    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            push!(jobs, (fock_bra, config_bra, coeff_bra, zeros(T,R)) )
        end
    end

    function _add_val!(eval_job, me, coeff_bra, coeff_ket)
        for ri in 1:R
            #for rj in ri:R
            #    @inbounds eval_job[ri,rj] += me * coeff_bra[ri] * coeff_ket[rj] 
            #    #eval_job[rj,ri] = eval_job[ri,rj]
            #end
            @inbounds eval_job[ri] += me * coeff_bra[ri] * coeff_ket[ri] 
        end
    end

    function do_job(job)
        fock_bra = job[1]
        config_bra = job[2]
        coeff_bra = job[3]
        eval_job = job[4]
        ket_idx = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            if haskey(clustered_ham, fock_trans) == false
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, coeff_ket) in configs_ket
                #ket_idx += 1
                #ket_idx <= job[1] || continue

                me = 0.0
                for term in clustered_ham[fock_trans]

                    #length(term.clusters) <= 2 || continue
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                    me += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    #Hrow[ket_idx] += me 
                    #H[job[1],ket_idx] += me 
                end
                #
                # now add the results
                #@inbounds for ri in 1:R
                #    @simd for rj in ri:R
                _add_val!(eval_job, me, coeff_bra, coeff_ket)
                #for ri in 1:R
                #    for rj in ri:R
                #        eval_job[ri,rj] += me * coeff_bra[ri] * coeff_ket[rj] 
                #        #eval_job[rj,ri] = eval_job[ri,rj]
                #    end
                #end
            end
        end
    end

    #for job in jobs
    #Threads.@threads for job in jobs
    @qthreads for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    for job in jobs
        evals .+= job[4]
    end

    return evals 
end
#=}}}=#




"""
    open_matvec(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.
"""
function open_matvec(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    println(" In open_matvec")
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    #sig = ClusteredState(clusters)
    #sig = OrderedDict{FockConfig{N}, OrderedDict{NTuple{N,Int16}, MVector{T} }}()
    for (fock_ket, configs_ket) in ci_vector.data
        for (ftrans, terms) in clustered_ham
            fock_bra = ftrans + fock_ket
           
            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_bra) || continue 
            all(f[2] >= 0 for f in fock_bra) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
        
            #if haskey(sig, fock_bra) == false
            #    sig[fock_bra] = OrderedDict{NTuple{N,Int16}, MVector{T}}()
            #end
            haskey(sig, fock_bra) || add_fockconfig!(sig, fock_bra)
            for term in terms

                length(term.clusters) <= nbody || continue

                for (config_ket, coeff_ket) in configs_ket
                    
                    sig_i = contract_matvec(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, thresh=thresh)
                    #if term isa ClusteredTerm2B
                    #    @btime sig_i = contract_matvec($term, $cluster_ops, $fock_bra, $fock_ket, $config_ket, $coeff_ket, thresh=$thresh)
                    #end
                    #typeof(sig_i) == typeof(sig[fock_bra]) || println(typeof(sig_i), "\n",  typeof(sig[fock_bra]), "\n")
                    
                    merge!(+, sig[fock_bra], sig_i)
                    #sig[fock_bra] = merge(+, sig[fock_bra], sig_i)
                    
                    #for (config,coeff) in sig_i
                    #    #display(coeff[1])
                    #    #display(sig[fock_bra][config][1])
                    #    sig[fock_bra][config][1] += coeff[1]
                    #    #sig[fock_bra][config] = sig[fock_bra][config] + coeff
                    #end
                end
            end
        end
    end

    return sig
end
#=}}}=#


"""
    open_matvec_thread(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.

This parallellizes over FockConfigs in the output state, so it's not the most fine-grained, but it avoids data races in 
filling the final vector
"""
function open_matvec_thread(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    println(" In open_matvec_thread")
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    #sig = ClusteredState(clusters)
    #sig = OrderedDict{FockConfig{N}, OrderedDict{NTuple{N,Int16}, MVector{T} }}()

    for (fock_ket, configs_ket) in ci_vector.data
        for (ftrans, terms) in clustered_ham
            fock_bra = ftrans + fock_ket

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_bra) || continue 
            all(f[2] >= 0 for f in fock_bra) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
           
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_bra)
                push!(jobs[fock_bra], job_input)
            else
                jobs[fock_bra] = [job_input]
            end
            
        end
    end

    jobs_vec = []
    for (fock_bra, job) in jobs
        push!(jobs_vec, (fock_bra, job))
    end

    jobs_out = Vector{ClusteredState{T,N,R}}()
    for tid in 1:Threads.nthreads()
        push!(jobs_out, ClusteredState(clusters, T=T, R=R))
    end


    #println(" Number of jobs:    ", length(jobs))
    #println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    #Threads.@threads for job in jobs_vec
   

    #for job in jobs_vec
    #@qthreads for job in jobs_vec
    Threads.@threads for job in jobs_vec
        fock_bra = job[1]
        sigi = _open_matvec_job(job[2], fock_bra, cluster_ops, nbody, thresh, N, R, T)
        tmp = jobs_out[Threads.threadid()]
        jobs_out[Threads.threadid()][fock_bra] = sigi
    end

    for threadid in 1:Threads.nthreads()
        #display(size(jobs_out[threadid]))
        add!(sig, jobs_out[threadid])
    end

    #BLAS.set_num_threads(Threads.nthreads())
    return sig
end
#=}}}=#

function _open_matvec_job(job, fock_bra, cluster_ops, nbody, thresh, N, R, T)
#={{{=#
    sigfock = OrderedDict{ClusterConfig{N}, MVector{R, T} }()

    for jobi in job 

        terms, fock_ket, configs_ket = jobi

        for term in terms

            length(term.clusters) <= nbody || continue

            for (config_ket, coeff_ket) in configs_ket

                sig_i = contract_matvec(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, thresh=thresh)
                #if term isa ClusteredTerm2B
                #    @btime sig_i = contract_matvec($term, $cluster_ops, $fock_bra, $fock_ket, $config_ket, $coeff_ket, thresh=$thresh)
                #    error("here")
                #end
                merge!(+, sigfock, sig_i)
            end
        end
    end
    return sigfock
end
#=}}}=#

"""
    open_matvec_parallel(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.

This parallellizes over FockConfigs in the output state, so it's not the most fine-grained, but it avoids data races in 
filling the final vector
"""
function open_matvec_parallel(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
   
    println(" Copy data to each worker")
    @sync for pid in procs()
        @spawnat pid eval(:(ci_vector = deepcopy($ci_vector)))
        @spawnat pid eval(:(sig_job = ClusteredState($clusters, R=$R)))
        @spawnat pid eval(:(cluster_ops = $cluster_ops))
        @spawnat pid eval(:(clusters = $clusters))
        @spawnat pid eval(:(thresh = $thresh))
    end
    flush(stdout)

    println(" Collect jobs")
    @time for (fock_ket, configs_ket) in ci_vector.data
        for (ftrans, terms) in clustered_ham
            fock_bra = ftrans + fock_ket

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_bra) || continue 
            all(f[2] >= 0 for f in fock_bra) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
           
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_bra)
                push!(jobs[fock_bra], job_input)
            else
                jobs[fock_bra] = [job_input]
            end
            
        end
    end

    jobs_vec = []
    for (fock_bra, job) in jobs
        push!(jobs_vec, (fock_bra, job))
    end

    jobs_out = Dict{Int, ClusteredState{T,N,R}}()
    for pid in procs()
        jobs_out[pid] = ClusteredState(clusters, T=T, R=R)
    end


    println(" Number of jobs:    ", length(jobs))
   

    #@sync @distributed for job in jobs_vec
   
    futures = []

    println(" Compute all jobs")
    @time @sync begin
        for job in jobs_vec
            fock_bra = job[1]
            future_sigi = @spawnat :any _open_matvec_job_parallel(job[2], fock_bra, nbody, thresh, N, R, T)
            #jobs_out[myid()][fock_bra] = sigi
            push!(futures, future_sigi)
        end
    end

    println(" Combine results")
    flush(stdout)
    @time for pid in procs()
        add!(sig, @fetchfrom pid sig_job)
    end

    #BLAS.set_num_threads(Threads.nthreads())
    return sig
end
#=}}}=#

function _open_matvec_job_parallel(job, fock_bra, nbody, thresh, N, R, T)
#={{{=#
    #sigfock = OrderedDict{ClusterConfig{N}, MVector{R, T} }()

    #sig = ClusteredState(clusters,R=R)
    add_fockconfig!(sig_job, fock_bra)

    for jobi in job 

        terms, fock_ket, configs_ket = jobi

        for term in terms

            length(term.clusters) <= nbody || continue

            for (config_ket, coeff_ket) in configs_ket

                sig_i = contract_matvec(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, thresh=thresh)
                #if term isa ClusteredTerm4B
                #    @btime sig_i = contract_matvec($term, $cluster_ops, $fock_bra, $fock_ket, $config_ket, $coeff_ket, thresh=$thresh)
                #    error("here")
                #end
                merge!(+, sig_job[fock_bra], sig_i)
            end
        end
    end
    return 
end
#=}}}=#

"""
    open_matvec_parallel2(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
"""
function open_matvec_parallel2(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters

    println(" create empty sig vector on each worker")
    @sync for pid in procs()
        @spawnat pid eval(:(ci_vector = deepcopy($ci_vector)))
        @spawnat pid eval(:(sig_job = ClusteredState($clusters, R=$R)))
        @spawnat pid eval(:(cluster_ops = $cluster_ops))
        @spawnat pid eval(:(clusters = $clusters))
        @spawnat pid eval(:(thresh = $thresh))
    end
    println("done")
    flush(stdout)


    #jobs = Vector{Tuple{TransferConfig{N}, ClusteredTerm}}()
    #println(" Number of jobs: ", length(jobs))

   
    println(" Compute all jobs")
    futures = []
    @time @sync for (ftrans, terms) in clustered_ham
        for term in terms
            length(term.clusters) <= nbody || continue

            future = @spawnat :any _do_job(ftrans, term)
            #future = do_job(ftrans, term)
            push!(futures, future)
        end
    end

    println(" combine results")
    flush(stdout)
    @time @sync for pid in procs()
        add!(sig, @fetchfrom pid sig_job)
    end

#    n = length(futures)
#    @elapsed while n > 0 # print out results
#        add!(sig, take!(futures))
#        n = n - 1
#    end

    return sig
end
#=}}}=#

function _do_job(ftrans, term)
    for (fock_ket, configs_ket) in ci_vector.data
        fock_bra = ftrans + fock_ket

        #
        # check to make sure this fock config doesn't have negative or too many electrons in any cluster
        all(f[1] >= 0 for f in fock_bra) || continue 
        all(f[2] >= 0 for f in fock_bra) || continue 
        all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
        all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 

        haskey(sig_job, fock_bra) || add_fockconfig!(sig_job, fock_bra)


        for (config_ket, coeff_ket) in configs_ket

            sig_i = contract_matvec(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, thresh=thresh)

            merge!(+, sig_job[fock_bra], sig_i)
        end
    end
    return 
end


"""
    compute_diagonal(vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}

Form the diagonal of the hamiltonan, `clustered_ham`, in the basis defined by `vector`
"""
function compute_diagonal(vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}
    #={{{=#
    Hd = zeros(size(vector)[1])
    idx = 0
    zero_trans = TransferConfig([(0,0) for i in 1:N])
    for (fock_bra, configs_bra) in vector.data
        for (config_bra, coeff_bra) in configs_bra
            idx += 1
            for term in clustered_ham[zero_trans]
		    try
			    Hd[idx] += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_bra, config_bra)
		    catch
			    display(term)
			    display(fock_bra)
			    display(config_bra)
			    error()
		    end

            end
        end
    end
    return Hd
end
#=}}}=#


"""
    compute_diagonal!(Hd, vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}

Form the diagonal of the hamiltonan, `clustered_ham`, in the basis defined by `vector`
"""
function compute_diagonal!(Hd, vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}
    #={{{=#
    idx = 0
    zero_trans = TransferConfig([(0,0) for i in 1:N])
    for (fock_bra, configs_bra) in vector.data
        for (config_bra, coeff_bra) in configs_bra
            idx += 1
            for term in clustered_ham[zero_trans]
		    try
			    Hd[idx] += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_bra, config_bra)
		    catch
			    display(term)
			    display(fock_bra)
			    display(config_bra)
			    error()
		    end

            end
        end
    end
    return
end
#=}}}=#


"""
    expand_each_fock_space!(s::ClusteredState{T,N,R}, bases::Vector{ClusterBasis}) where {T,N,R}

For each fock space sector defined, add all possible basis states
- `basis::Vector{ClusterBasis}` 
"""
function expand_each_fock_space!(s::ClusteredState{T,N,R}, bases::Vector{ClusterBasis}) where {T,N,R}
    # {{{
    println("\n Make each Fock-Block the full space")
    # create full space for each fock block defined
    for (fblock,configs) in s.data
        #println(fblock)
        dims::Vector{UnitRange{Int16}} = []
        #display(fblock)
        for c in s.clusters
            # get number of vectors for current fock space
            dim = size(bases[c.idx][fblock[c.idx]], 2)
            push!(dims, 1:dim)
        end
        for newconfig in product(dims...)
            #display(newconfig)
            #println(typeof(newconfig))
            #
            # this is not ideal - need to find a way to directly create key
            config = ClusterConfig(collect(newconfig))
            s.data[fblock][config] = zeros(SVector{R,T}) 
            #s.data[fblock][[i for i in newconfig]] = 0
        end
    end
end
# }}}

"""
    expand_to_full_space!(s::AbstractState, bases::Vector{ClusterBasis}, na, nb)

Define all possible fock space sectors and add all possible basis states
- `basis::Vector{ClusterBasis}` 
- `na`: Number of alpha electrons total
- `nb`: Number of alpha electrons total
"""
function expand_to_full_space!(s::AbstractState, bases::Vector{ClusterBasis}, na, nb)
    # {{{
    println("\n Expand to full space")
    ns = []

    for c in s.clusters
        nsi = []
        for (fspace,basis) in bases[c.idx]
            push!(nsi,fspace)
        end
        push!(ns,nsi)
    end
    for newfock in product(ns...)
        nacurr = 0
        nbcurr = 0
        for c in newfock
            nacurr += c[1]
            nbcurr += c[2]
        end
        if (nacurr == na) && (nbcurr == nb)
            config = FockConfig(collect(newfock))
            add_fockconfig!(s,config) 
        end
    end
    expand_each_fock_space!(s,bases)

    return
end
# }}}




"""
    project_out!(v::ClusteredState, w::ClusteredState)

Project w out of v 
    |v'> = |v> - |w><w|v>
"""
function project_out!(v::ClusteredState, w::ClusteredState)
    for (fock,configs) in v.data 
        for (config, coeff) in configs
            if haskey(w, fock)
                if haskey(w[fock], config)
                    delete!(v.data[fock], config)
                end
            end
        end
    end
end



"""
    hosvd(ci_vector::ClusteredState{T,N,R}, cluster_ops; hshift=1e-8, truncate=-1) where {T,N,R}

Peform HOSVD aka Tucker Decomposition of ClusteredState
"""
function hosvd(ci_vector::ClusteredState{T,N,R}, cluster_ops; hshift=1e-8, truncate=-1) where {T,N,R}
#={{{=#
   
    cluster_rotations = []
    for ci in ci_vector.clusters
        println()
        println(" --------------------------------------------------------")
        println(" Density matrix: Cluster ", ci.idx)
        println()
        println(" Compute BRDM")
        println(" Hshift = ",hshift)
        
        dims = Dict()
        for (fock, mat) in cluster_ops[ci.idx]["H"]
            fock[1] == fock[2] || error("?")
            dims[fock[1]] = size(mat,1)
        end
        
        rdms = build_brdm(ci_vector, ci, dims)
        norm = 0
        entropy = 0
        rotations = Dict{Tuple,Matrix{T}}() 
        for (fspace,rdm) in rdms
            fspace_norm = 0
            fspace_entropy = 0
            @printf(" Diagonalize RDM for Cluster %2i in Fock space: ",ci.idx)
            println(fspace)
            F = eigen(Symmetric(rdm))

            idx = sortperm(F.values, rev=true) 
            n = F.values[idx]
            U = F.vectors[:,idx]


            # Either truncate the unoccupied cluster states, or remix them with a hamiltonian to be unique
            if truncate < 0
                remix = []
                for ni in 1:length(n)
                    if n[ni] < 1e-8
                        push!(remix, ni)
                    end
                end
                U2 = U[:,remix]
                Hlocal = U2' * cluster_ops[ci.idx]["H"][(fspace,fspace)] * U2
                
                F = eigen(Symmetric(Hlocal))
                n2 = F.values
                U2 = U2 * F.vectors
                
                U[:,remix] .= U2[:,:]
            
            else
                keep = []
                for ni in 1:length(n) 
                    if abs(n[ni]) > truncate
                        push!(keep, ni)
                    end
                end
                @printf(" Truncated Tucker space. Starting: %5i Ending: %5i\n" ,length(n), length(keep))
                U = U[:,keep]
            end
        

           
            
            n = diag(U' * rdm * U)
            Elocal = diag(U' * cluster_ops[ci.idx]["H"][(fspace,fspace)] * U)
            
            norm += sum(n)
            fspace_norm = sum(n)
            @printf("                 %4s:    %12s    %12s\n", "","Population","Energy")
            for (ni_idx,ni) in enumerate(n)
                if abs(ni/norm) > 1e-16
                    fspace_entropy -= ni*log(ni/norm)/norm
                    entropy -=  ni*log(ni)
                    @printf("   Rotated State %4i:    %12.8f    %12.8f\n", ni_idx,ni,Elocal[ni_idx])
                end
           end
           @printf("   ----\n")
           @printf("   Entanglement entropy:  %12.8f\n" ,fspace_entropy) 
           @printf("   Norm:                  %12.8f\n" ,fspace_norm) 

           #
           # let's just be careful that our vectors remain orthogonal
           F = svd(U)
           U = F.U * F.Vt
           check_orthogonality(U) 
           rotations[fspace] = U
        end
        @printf(" Final entropy:.... %12.8f\n",entropy)
        @printf(" Final norm:....... %12.8f\n",norm)
        @printf(" --------------------------------------------------------\n")

        flush(stdout) 

        #ci.rotate_basis(rotations)
        #ci.check_basis_orthogonality()
        push!(cluster_rotations, rotations)
    end
    return cluster_rotations
end
#=}}}=#




"""
    build_brdm(ci_vector::ClusteredState, ci, dims)
    
Build block reduced density matrix for `Cluster`,  `ci`
- `ci_vector::ClusteredState` = input state
- `ci` = Cluster type for whihch we want the BRDM
- `dims` = list of dimensions for each fock sector
"""
function build_brdm(ci_vector::ClusteredState, ci, dims)
    # {{{
    rdms = OrderedDict()
    for (fspace, configs) in ci_vector.data
        curr_dim = dims[fspace[ci.idx]]
        rdm = zeros(curr_dim,curr_dim)
        for (configi,coeffi) in configs
            for cj in 1:curr_dim

                configj = [configi...]
                configj[ci.idx] = cj
                configj = ClusterConfig(configj)

                if haskey(configs, configj)
                    rdm[configi[ci.idx],cj] += sum(coeffi.*configs[configj])
                end
            end
        end


        if haskey(rdms, fspace[ci.idx]) 
            rdms[fspace[ci.idx]] += rdm 
        else
            rdms[fspace[ci.idx]] = rdm 
        end

    end
    return rdms
end
# }}}



function dump_tpsci(filename::AbstractString, ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    @save filename ci_vector cluster_ops clustered_ham
end

#function load_tpsci(filename::AbstractString) 
#    a = @load filename
#    return eval.(a)
#end

