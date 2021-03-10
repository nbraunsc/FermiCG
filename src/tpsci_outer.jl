#"""
#    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#
#Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
#"""
#function build_full_H_serial(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
##={{{=#
#    dim = length(ci_vector)
#    H = zeros(dim, dim)
#
#    zero_fock = FermiCG.TransferConfig([(0,0) for i in ci_vector.clusters])
#    bra_idx = 0
#    for (fock_bra, configs_bra) in ci_vector.data
#        for (config_bra, coeff_bra) in configs_bra
#            bra_idx += 1
#            ket_idx = 0
#            for (fock_ket, configs_ket) in ci_vector.data
#                fock_trans = fock_bra - fock_ket
#
#                # check if transition is connected by H
#                if haskey(clustered_ham, fock_trans) == false
#                    ket_idx += length(configs_ket)
#                    continue
#                end
#
#                for (config_ket, coeff_ket) in configs_ket
#                    ket_idx += 1
#                    ket_idx <= bra_idx || continue
#
#
#                    for term in clustered_ham[fock_trans]
#                    
#                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
#                        
#                        me = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
#                        H[bra_idx, ket_idx] += me 
#                    end
#
#                    H[ket_idx, bra_idx] = H[bra_idx, ket_idx]
#
#                end
#            end
#        end
#    end
#    return H
#end
##=}}}=#


"""
    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    jobs = []

    zero_fock = FermiCG.TransferConfig([(0,0) for i in ci_vector.clusters])
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
                        
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    
                    me = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
#                    if term isa ClusteredTerm2B
#                        @btime FermiCG.contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
#                        error("huh")
#                    end
                    Hrow[ket_idx] += me 
                    #H[job[1],ket_idx] += me 
                end

            end

        end
    end

    # because @threads divides evenly the loop, let's distribute thework more fairly
    mid = length(jobs) ÷ 2
    r = collect(1:length(jobs))
    perm = [r[1:mid] reverse(r[mid+1:end])]'[:]
    #display(perm)
    jobs = jobs[perm]
    Threads.@threads for job in jobs
    #Threads.@threads for job in shuffle(jobs)
    #for job in jobs
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


function matvec(ci_vector::ClusteredState, cluster_ops, clustered_ham)
end
