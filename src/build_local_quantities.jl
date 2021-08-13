"""
    get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)

For a given `ClusterSubspace`, `tss`, return the subspace remaining
"""
function get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)
#={{{=#
    data = OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}()
    for (fock,basis) in cb
    
        if haskey(tss.data,fock)
            first(tss.data[fock]) == 1 || error(" p-space doesn't include ground state?")
            newrange = last(tss[fock])+1:size(cb[fock],2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        else
            newrange = 1:size(cb[fock],2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        end
    end

    return ClusterSubspace(tss.cluster, data)
#=}}}=#
end



"""
    compute_cluster_ops(cluster_bases::Vector{ClusterBasis})
"""
function compute_cluster_ops(cluster_bases::Vector{ClusterBasis},ints; T=Float64)
#={{{=#
    clusters = Vector{Cluster}()
    for ci in cluster_bases
        push!(clusters, ci.cluster)
    end
    
    cluster_ops = Vector{ClusterOps{T,3}}()
    cluster_ops_local = Vector{ClusterOps{T,2}}()
    for ci in clusters
        push!(cluster_ops, ClusterOps(ci,3)) 
        push!(cluster_ops_local, ClusterOps(ci,2)) 
    end


    for ci in clusters
        cb = cluster_bases[ci.idx]
        
        cluster_ops_local[ci.idx]["H"] = FermiCG.tdm_H(cb, subset(ints, ci.orb_list), verbose=0) 

        cluster_ops[ci.idx]["A"], cluster_ops[ci.idx]["a"] = FermiCG.tdm_A(cb,"alpha") 
        cluster_ops[ci.idx]["B"], cluster_ops[ci.idx]["b"] = FermiCG.tdm_A(cb,"beta")
        cluster_ops[ci.idx]["AA"], cluster_ops[ci.idx]["aa"] = FermiCG.tdm_AA(cb,"alpha") 
        cluster_ops[ci.idx]["BB"], cluster_ops[ci.idx]["bb"] = FermiCG.tdm_AA(cb,"beta") 
        cluster_ops[ci.idx]["Aa"] = FermiCG.tdm_Aa(cb,"alpha") 
        cluster_ops[ci.idx]["Bb"] = FermiCG.tdm_Aa(cb,"beta") 
        cluster_ops[ci.idx]["Ab"], cluster_ops[ci.idx]["Ba"] = FermiCG.tdm_Ab(cb) 
        # remove BA and ba account for these terms 
        cluster_ops[ci.idx]["AB"], cluster_ops[ci.idx]["ba"], cluster_ops[ci.idx]["BA"], cluster_ops[ci.idx]["ab"] = FermiCG.tdm_AB(cb)
        cluster_ops[ci.idx]["AAa"], cluster_ops[ci.idx]["Aaa"] = FermiCG.tdm_AAa(cb,"alpha")
        cluster_ops[ci.idx]["BBb"], cluster_ops[ci.idx]["Bbb"] = FermiCG.tdm_AAa(cb,"beta")
        cluster_ops[ci.idx]["ABa"], cluster_ops[ci.idx]["Aba"] = FermiCG.tdm_ABa(cb,"alpha")
        cluster_ops[ci.idx]["ABb"], cluster_ops[ci.idx]["Bba"] = FermiCG.tdm_ABa(cb,"beta")
        #cluster_ops[ci.idx]["ABa"], cluster_ops[ci.idx]["Aba"], cluster_ops[ci.idx]["BAa"], cluster_ops[ci.idx]["Aab"] = FermiCG.tdm_ABa(cb,"alpha")
        #cluster_ops[ci.idx]["ABb"], cluster_ops[ci.idx]["Bba"], cluster_ops[ci.idx]["BAb"], cluster_ops[ci.idx]["Bab"] = FermiCG.tdm_ABa(cb,"beta")
       
        # spin operators
        
        #
        # S+
        op = Dict{Tuple{FockIndex,FockIndex},Array{T,3}}()
        for (fock,mat) in cluster_ops[ci.idx]["Ab"]
            mat2 = reshape(mat, (length(ci), length(ci), size(mat,2), size(mat,3)))
            dims = size(mat2)

            op[fock] = zeros(length(ci), size(mat,2), size(mat,3))
            for j in 1:dims[4]
                for i in 1:dims[3]
                    for p in 1:dims[2]
                        op[fock][p,i,j] = mat2[p,p,i,j]
                    end
                end
            end
        end
        cluster_ops[ci.idx]["S+"] = op
        
        #
        # S-
        op = Dict{Tuple{FockIndex,FockIndex},Array{T,3}}()
        for (fock,mat) in cluster_ops[ci.idx]["Ba"]
            mat2 = reshape(mat, (length(ci), length(ci), size(mat,2), size(mat,3)))
            dims = size(mat2)

            op[fock] = zeros(length(ci), size(mat,2), size(mat,3))
            for j in 1:dims[4]
                for i in 1:dims[3]
                    for p in 1:dims[2]
                        op[fock][p,i,j] = mat2[p,p,i,j]
                    end
                end
            end
        end
        cluster_ops[ci.idx]["S-"] = op
        
        #
        # Sz
        op = Dict{Tuple,Array}()
        #
        # loop over fock-space transitions
        for (fock,basis) in cb
            focktrans = (fock,fock)

            sz = (fock[1] - fock[2]) / 2.0
            op[focktrans] = sz*Matrix(1.0I, size(cb[fock],2), size(cb[fock],2))
            op[focktrans] = reshape(op[focktrans],1,size(op[focktrans],1),size(op[focktrans],2))

        end
        cluster_ops[ci.idx]["Sz"] = op


        #
        # S2
        cluster_ops_local[ci.idx]["S2"] = FermiCG.tdm_S2(cb, subset(ints, ci.orb_list), verbose=0) 

        # Compute single excitation operator
        tmp = Dict{Tuple,Array}()
        for (fock,basis) in cb
            tmp[(fock,fock)] = (cluster_ops[ci.idx]["Aa"][(fock,fock)] + cluster_ops[ci.idx]["Bb"][(fock,fock)])
        end
        cluster_ops[ci.idx]["E1"] = tmp 

#        #
#        # reshape data into 3index quantities: e.g., (pqr, I, J)
#        for opstring in keys(cluster_ops[ci.idx])
#            #opstring != "H" || continue
#            #opstring != "S2" || continue
#            for ftrans in keys(cluster_ops[ci.idx][opstring])
#                data = cluster_ops[ci.idx][opstring][ftrans]
#                dim1 = prod(size(data)[1:(length(size(data))-2)])
#                dim2 = size(data)[length(size(data))-1]
#                dim3 = size(data)[length(size(data))-0]
#                cluster_ops[ci.idx][opstring][ftrans] = copy(reshape(data, (dim1,dim2,dim3)))
#            end
#        end
    end
    return cluster_ops, cluster_ops_local
end
    #=}}}=#


"""
    tdm_H(cb::ClusterBasis; verbose=0)

Compute local Hamiltonian `<s|H|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_H(cb::ClusterBasis{T}, ints; verbose=0) where T
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple{FockIndex, FockIndex},Array{T,2}}()
    #
    # loop over fock-space transitions
    verbose == 0 || display(cb.cluster)
    for (fock,basis) in cb
        focktrans = (fock,fock)
        problem = StringCI.FCIProblem(norbs, fock[1], fock[2])
        verbose == 0 || display(problem)
        Hmap = StringCI.get_map(ints, problem)

        H = cb[fock]' * Matrix((Hmap * cb[fock]))
        dicti[focktrans] = H

        if verbose > 0
            for e in 1:size(cb[fock],2)
                @printf(" %4i %12.8f\n", e, dicti[focktrans][e,e])
            end
        end
    end
    return dicti
#=}}}=#
end


"""
"""
function tdm_S2(cb::ClusterBasis{T}, ints; verbose=0) where T
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple{FockIndex, FockIndex}, Array{T,2}}()
    #
    # loop over fock-space transitions
    verbose == 0 || display(cb.cluster)
    for (fock,basis) in cb
        focktrans = (fock,fock)
        problem = StringCI.FCIProblem(norbs, fock[1], fock[2])
        verbose == 0 || display(problem)

        op = cb[fock]' * (StringCI.build_S2_matrix(problem) * cb[fock])
        
        dicti[focktrans] = op

        if verbose > 0
            for e in 1:size(cb[fock],2)
                @printf(" %4i %12.8f\n", e, dicti[focktrans][e,e])
            end
        end
    end
    return dicti
#=}}}=#
end


"""
    tdm_A(cb::ClusterBasis; verbose=0)

Compute `<s|p'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_A(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na+1,nb)
            elseif spin_case == "beta"
                fockbra = (na,nb+1)
            else
                throw(DomainError(spin_case))
            end
            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_annihilation(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [1,3,2])
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_AA(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AA(cb::ClusterBasis{T}, spin_case; verbose=0) where T
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dicti_adj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na+2,nb)
            elseif spin_case == "beta"
                fockbra = (na,nb+2)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                #println()
                #println("::::::::::::::: ", fockbra, fockket)
                #println(":: ", size(basis_bra), size(basis_ket))
                
                op = FermiCG.StringCI.compute_AA(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                op_t =  permutedims(op, [2,1,4,3])

                dicti[focktrans] = convert(Array,reshape(op, (size(op,1)*size(op,2), size(op,3), size(op,4))))
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] = convert(Array,reshape(op_t, (size(op_t,1)*size(op_t,2), size(op_t,3), size(op_t,4))))
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Aa(cb::ClusterBasis{T}, spin_case; verbose=0) where T
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = (na,nb)

            fockket = (na,nb)
            focktrans = (fockbra,fockket)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                op = FermiCG.StringCI.compute_Aa(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                dicti[focktrans] = convert(Array,reshape(op, (size(op,1)*size(op,2), size(op,3), size(op,4))))
                #dicti[focktrans] = reshape(dicti[focktrans],(norbs*norbs, size(dicti[focktrans],3), size(dicti[focktrans],4)))
            end
        end
    end
    return dicti
#=}}}=#
end


"""
    tdm_Ab(cb::ClusterBasis; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Ab(cb::ClusterBasis{T}, verbose=0) where T
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dicti_adj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    #
    # loop over fock-space transitions
    for na in -1:norbs+1
        for nb in -1:norbs+1
            fockbra = (na+1,nb-1)

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                op = FermiCG.StringCI.compute_Ab(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                op_t =  permutedims(op, [2,1,4,3])
                dicti[focktrans] = convert(Array,reshape(op, (size(op,1)*size(op,2), size(op,3), size(op,4))))
                
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                #dicti[focktrans] = FermiCG.StringCI.compute_Ba(norbs, fockket[1], fockket[2], fockket[1], fockket[2], basis_bra, basis_ket)
                dicti_adj[focktrans_adj] = convert(Array,reshape(op_t, (size(op_t,1)*size(op_t,2), size(op_t,3), size(op_t,4))))
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_AB(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q'` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AB(cb::ClusterBasis{T}; verbose=0) where T
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dicti_adj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dictj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dictj_adj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    #
    # loop over fock-space transitions
    for na in -2:norbs+2
        for nb in -2:norbs+2
            fockbra = (na+1,nb+1)

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                opi = FermiCG.StringCI.compute_AB(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                opj = -permutedims(opi, [2,1,3,4])
                dicti[focktrans] = convert(Array,reshape(opi, (size(opi,1)*size(opi,2), size(opi,3), size(opi,4))))
                dictj[focktrans] = convert(Array,reshape(opj, (size(opj,1)*size(opj,2), size(opj,3), size(opj,4))))
                
                # adjoint 
                opi_t =  permutedims(opi, [2,1,4,3])
                opj_t =  permutedims(opj, [2,1,4,3])
                
                dicti_adj[focktrans_adj] = convert(Array,reshape(opi_t, (size(opi_t,1)*size(opi_t,2), size(opi_t,3), size(opi_t,4))))
                dictj_adj[focktrans_adj] = convert(Array,reshape(opj_t, (size(opj_t,1)*size(opj_t,2), size(opj_t,3), size(opj_t,4))))
            end
        end
    end
    return dicti, dicti_adj, dictj, dictj_adj
#=}}}=#
end


"""
    tdm_AAa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q'r|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AAa(cb::ClusterBasis{T}, spin_case; verbose=0) where T
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dicti_adj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na+1,nb)
            elseif spin_case == "beta"
                fockbra = (na,nb+1)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                op = FermiCG.StringCI.compute_AAa(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                op_t =  permutedims(op, [3,2,1,5,4])
                
                dicti[focktrans] = convert(Array,reshape(op, (size(op,1)*size(op,2)*size(op,3), 
                                                              size(op,4), size(op,5))))
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] = convert(Array,reshape(op_t, 
                                                             (size(op_t,1)*size(op_t,2)*size(op_t,3), 
                                                              size(op_t,4), size(op_t,5))))
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_ABa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q'r|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_ABa(cb::ClusterBasis{T}, spin_case; verbose=0) where T
    #={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dicti_adj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dictj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    dictj_adj = Dict{Tuple{FockIndex, FockIndex},Array{T,3}}()
    #
    # loop over fock-space transitions
    for na in -2:norbs+2
        for nb in -2:norbs+2
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na,nb+1)
            elseif spin_case == "beta"
                fockbra = (na+1,nb)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                if spin_case == "alpha"
                    opi = FermiCG.StringCI.compute_ABa(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                    opj = -permutedims(opi, [2,1,3,4,5])
                    
                    opi_t =  permutedims(opi, [3,2,1,5,4])
                    opj_t =  permutedims(opj, [3,2,1,5,4])
                    dicti[focktrans] = convert(Array,reshape(opi, (size(opi,1)*size(opi,2)*size(opi,3), 
                                                                  size(opi,4), size(opi,5))))
                    dictj[focktrans] = convert(Array,reshape(opj, (size(opj,1)*size(opj,2)*size(opj,3), 
                                                                  size(opj,4), size(opj,5))))
                    # adjoint 
                    basis_bra = cb[fockket]
                    basis_ket = cb[fockbra]
                    dicti_adj[focktrans_adj] = convert(Array,reshape(opi_t, 
                                                                 (size(opi_t,1)*size(opi_t,2)*size(opi_t,3), 
                                                                  size(opi_t,4), size(opi_t,5))))
                    dictj_adj[focktrans_adj] = convert(Array,reshape(opj_t, 
                                                                 (size(opj_t,1)*size(opj_t,2)*size(opj_t,3), 
                                                                  size(opj_t,4), size(opj_t,5))))
                elseif spin_case == "beta"
                    opi = FermiCG.StringCI.compute_ABb(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                    opj = -permutedims(opi, [2,1,3,4,5])
                    opi_t =  permutedims(opi, [3,2,1,5,4])
                    opj_t =  permutedims(opj, [3,2,1,5,4])
                    dicti[focktrans] = convert(Array,reshape(opi, (size(opi,1)*size(opi,2)*size(opi,3), 
                                                                  size(opi,4), size(opi,5))))
                    dictj[focktrans] = convert(Array,reshape(opj, (size(opj,1)*size(opj,2)*size(opj,3), 
                                                                  size(opj,4), size(opj,5))))
                    # adjoint 
                    basis_bra = cb[fockket]
                    basis_ket = cb[fockbra]
                    dicti_adj[focktrans_adj] = convert(Array,reshape(opi_t, 
                                                                 (size(opi_t,1)*size(opi_t,2)*size(opi_t,3), 
                                                                  size(opi_t,4), size(opi_t,5))))
                    dictj_adj[focktrans_adj] = convert(Array,reshape(opj_t, 
                                                                 (size(opj_t,1)*size(opj_t,2)*size(opj_t,3), 
                                                                  size(opj_t,4), size(opj_t,5))))
                else
                    error("Wrong spin_case: ",spin_case)
                end
            end
        end
    end
    return dicti, dicti_adj, dictj, dictj_adj
    #=}}}=#
end



"""
    function add_cmf_operators!(ops::Vector{ClusterOps}, bases::Vector{ClusterBasis}, ints, Da, Db; verbose=0)

Add effective local hamiltonians (local CASCI) type hamiltonians to a `ClusterOps` type for each `Cluster'
"""
function add_cmf_operators!(ops::Vector{ClusterOps{T,2}}, bases::Vector{ClusterBasis}, ints, Da, Db; verbose=0) where T
    #={{{=#
    n_clusters = length(bases)
    for ci_idx in 1:n_clusters
        cb = bases[ci_idx]
        ci = cb.cluster
        verbose == 0 || println()
        verbose == 0 || display(ci)
        norbs = length(cb.cluster)
        
        ints_i = form_casci_ints(ints, ci, Da, Db)


        dicti = Dict{Tuple{FockIndex, FockIndex},Array{T,2}}()
        
        #
        # loop over fock-space transitions
        verbose == 0 || display(cb.cluster)
        for (fock,basis) in cb
            focktrans = (fock,fock)
            problem = StringCI.FCIProblem(norbs, fock[1], fock[2])
            verbose == 0 || display(problem)
            Hmap = StringCI.get_map(ints_i, problem)

            dicti[focktrans] = cb[fock]' * Matrix((Hmap * cb[fock]))

            if verbose > 0
                for e in 1:size(cb[fock],2)
                    @printf(" %4i %12.8f\n", e, dicti[focktrans][e,e])
                end
            end
        end
        ops[ci.idx]["Hcmf"] = dicti
    end
    return 
end
#=}}}=#


"""
    function form_schmidt_basis(ints::InCoreInts, ci::Cluster, Da, Db; 
        thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6,do_embedding=true,
        eig_nr=1, eig_max_cycles=200)

thresh_orb      :   threshold for determining how many bath orbitals to include
thresh_schmidt  :   threshold for determining how many singular vectors to include for cluster basis

Returns new basis for the cluster
"""
function form_schmidt_basis(ints::InCoreInts, ci::Cluster, Da, Db; 
        thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6,do_embedding=true,
        eig_nr=1, eig_max_cycles=200)
#={{{=#
    println()
    println("------------------------------------------------------------")
    @printf("Form Embedded Schmidt-style basis for Cluster %4i\n",ci.idx)
    D = Da + Db 

    # Form the exchange matrix
    K = zeros(size(ints.h1))
    @tensor begin
	K[q,r]  = ints.h2[p,q,r,s] * D[p,s]
    end

    no = size(ints.h1,1)
    ci_no = length(ci.orb_list)


    na_tot = Int(round(tr(Da)))
    nb_tot = Int(round(tr(Db)))
    println(" Number of electrons in full system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_tot,nb_tot)

    active = ci.orb_list

    backgr = Vector{Int}()
    for i in 1:no
    	if !(i in active)
    	    append!(backgr,i)
        end
    end

    println("active",active)
    println("backgr",backgr)

    K2 = zeros((ci_no,no-ci_no))

    for (pi,p) in enumerate(active)
    	for (qi,q) in enumerate(backgr)
	    K2[pi,qi] = K[p,q]
	end
    end

    println("The exchange matrix:")
    display(K2)
    F = svd(K2,full=true)

    @printf("\nSing. Val.\n")
    nkeep = 0
    for si in F.S
    	@printf("%16.12f\n",si)
        if si > thresh_orb
            nkeep += 1
        end
    end

    C = zeros(size(ints.h1))
    for (pi,p) in enumerate(active)
    	for (qi,q) in enumerate(active)
	    if pi==qi
	        C[p,qi] = 1
	    end
	end
    end

    #display(C)

    #display(F.Vt)
    #display(F.U)
    for (pi,p) in enumerate(backgr)
    	for (qi,q) in enumerate(backgr)
	    C[p,qi+length(active)] = F.Vt[qi,pi]
	end
    end
    #display(C)

    Cfrag = C[:,1:ci_no]
    Cbath = C[:,ci_no+1:ci_no+nkeep]
    Cenvt = C[:,ci_no+nkeep+1:end]
    
    @printf("Cfrag\n")
    display(Cfrag)
    @printf("\n NElec: %12.8f\n",(tr(Cfrag'*(Da+Db)*Cfrag)))
    @printf("Cbath\n")
    display(Cbath)
    @printf("\n NElec: %12.8f\n",(tr(Cbath'*(Da+Db)*Cbath)))
    @printf("Cenv\n")
    display(Cenvt)
    @printf("\n NElec: %12.8f\n",(tr(Cenvt'*(Da+Db)*Cenvt)))

    K2 = C'* K * C
    Da2 = C'* Da * C
    Db2 = C'* Db * C

    na = tr(Da2[1:ci_no+nkeep,1:ci_no+nkeep])
    nb = tr(Db2[1:ci_no+nkeep,1:ci_no+nkeep])

    println(" Number of electrons in Fragment+Bath system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na,nb)

    denvt_a = Cenvt*Cenvt'*Da*Cenvt*Cenvt'
    denvt_b = Cenvt*Cenvt'*Db*Cenvt*Cenvt'
    #println(denvt_a)

    na_env = tr(denvt_a)
    nb_env = tr(denvt_b)

    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_env,nb_env)

    na_envt = Int(round(tr(Cenvt'*Da*Cenvt)))
    nb_envt = Int(round(tr(Cenvt'*Db*Cenvt)))


    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_envt,nb_envt)
    #display(Da)

    # rotate integrals to current subspace basis
    denvt_a = C'*denvt_a*C
    denvt_b = C'*denvt_b*C
    ints2 = orbital_rotation(ints, C)

    #avoid very zero numbers in diagonalization
    denvt_a[abs.(denvt_a) .< 1e-15] .= 0
    denvt_b[abs.(denvt_b) .< 1e-15] .= 0
    #display(denvt_a)
    #display(denvt_a)

    # find closest idempotent density for the environment
    if do_embedding
        if size(Cenvt,2)>0

	    #eigenvalue 
	    EIG = eigen(denvt_a)
	    U = EIG.vectors
	    n = EIG.values

	    U = U[:, sortperm(n,rev=true)]
	    n = n[sortperm(n,rev=true)]
	    #println(n)
	    #display(U)

            for i in 1:nkeep
                @assert(n[i]>1e-14)
	    end

            denvt_a = U[:,1:na_envt] * U[:,1:na_envt]'

	    EIG = eigen(denvt_b)
	    U = EIG.vectors
	    n = EIG.values

	    U = U[:, sortperm(n,rev=true)]
	    n = n[sortperm(n,rev=true)]

            for i in 1:nkeep
                @assert(n[i]>1e-14)
	    end

            denvt_b = U[:,1:nb_envt] * U[:,1:nb_envt]'

	end
    #form ints in the cluster 
    no_range = collect(1:size(Cfrag,2)+size(Cbath,2))

    #ints_f = subset(ints2,collect(1:size(Cfrag,2)+size(Cbath,2)), denvt_a, denvt_b)

    ints_f = FermiCG.form_1rdm_dressed_ints(ints2,no_range,denvt_a,denvt_b)

    else
        denvt_a *= 0 
        denvt_b *= 0 
        ints_f = form_casci_eff_ints(ints2,collect(1:size(Cfrag,2)+size(Cbath,2)), denvt_a, denvt_b)
    end

    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",tr(denvt_a),tr(denvt_b))

    na_actv = na_tot - na_envt
    nb_actv = nb_tot - nb_envt
    println(" Number of electrons in Fragment+Bath system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_actv,nb_actv)

    norb2 = size(ints_f.h1,1)
    problem = FermiCG.StringCI.FCIProblem(norb2, na_actv, nb_actv)
    Hmap = StringCI.get_map(ints_f, problem)
    v0 = svd(rand(problem.dim,eig_nr)).U
    davidson = FermiCG.Davidson(Hmap,v0=v0,max_iter=200, max_ss_vecs=20, nroots=eig_nr, tol=1e-8)
    #FermiCG.solve(davidson)
    @printf(" Now iterate: \n")
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    @time e,v = FermiCG.solve(davidson);
    e = real(e)[1]
    v = v[:,1]
    
    basis = FermiCG.StringCI.svd_state(v,problem,length(active),nkeep,thresh_schmidt)
    return basis
end
#=}}}=#


"""
    compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{Cluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing)

Return a Vector of `ClusterBasis` for each `Cluster` 
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{Cluster}`: Clusters 
- `verbose::Int`: Print level
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `delta_elec`: number of electrons different from reference (init_fspace)
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1a`: background density matrix for embedding local hamiltonian (alpha)
- `rdm1b`: background density matrix for embedding local hamiltonian (beta)
"""
function compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{Cluster}; 
                init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
                rdm1a=nothing, rdm1b=nothing)
#={{{=#
    # initialize output
    cluster_bases = Vector{ClusterBasis}()

    for ci in clusters
        verbose == 0 || display(ci)

        #
        # Get subset of integrals living on cluster, ci
        ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b) 

        if (rdm1a != nothing && init_fspace == nothing)
            error(" Cant embed withing init_fspace")
        end

        if all( (rdm1a,rdm1b,init_fspace) .!= nothing)
            # 
            # Verify that density matrix provided is consistent with reference fock sectors
            occs = diag(rdm1a)
            occs[ci.orb_list] .= 0
            na_embed = sum(occs)
            occs = diag(rdm1b)
            occs[ci.orb_list] .= 0
            nb_embed = sum(occs)
            verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f", na_embed, nb_embed)
        end
            
        delta_e_i = ()
        if all( (delta_elec,init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec)
        end
        
        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = FermiCG.possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci) 
        for sec in sectors
            
            #
            # prepare for FCI calculation for give sector of Fock space
            problem = FermiCG.StringCI.FCIProblem(length(ci), sec[1], sec[2])
            verbose == 0 || display(problem)
            nr = min(max_roots, problem.dim)

            #
            # Build full Hamiltonian matrix in cluster's Slater Det basis
            Hmat = FermiCG.StringCI.build_H_matrix(ints_i, problem)
            if problem.dim < 1000
                F = eigen(Hmat)

                e = F.values[1:nr]
                basis_i[sec] = F.vectors[:,1:nr]
                #display(e)
            else
                e,v = Arpack.eigs(Hmat, nev = nr, which=:SR)
                e = real(e)[1:nr]
                basis_i[sec] = v[:,1:nr]
            end
            if verbose > 0
                state=1
                for ei in e
                    @printf("   State %4i Energy: %12.8f %12.8f\n",state,ei, ei+ints.h0)
                    state += 1
                end
            end
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#


"""
    compute_cluster_est_basis(ints::InCoreInts, clusters::Vector{Cluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing)

Return a Vector of `ClusterBasis` for each `Cluster`  using the Embedded Schmidt Truncation
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{Cluster}`: Clusters 
- `Da`: background density matrix for embedding local hamiltonian (alpha)
- `Db`: background density matrix for embedding local hamiltonian (beta)
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `thresh_schmidt`: the threshold for the EST 
- `thresh_orb`: threshold for the orbital
- `thresh_ci`: threshold for the ci problem
"""
function compute_cluster_est_basis(ints::InCoreInts, clusters::Vector{Cluster},Da,Db; 
        thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6,
	do_embedding=true,verbose=0,init_fspace=nothing,delta_elec=nothing,
	est_nr=1, est_max_cycles=200, est_thresh=1e-6)
#={{{=#
    # initialize output
    cluster_bases = Vector{ClusterBasis}()

    for ci in clusters
        verbose == 0 || display(ci)

        # Obtain the schmidt basis
        basis = FermiCG.form_schmidt_basis(ints, ci, Da, Db,thresh_schmidt=thresh_schmidt,
					   eig_nr=est_nr, eig_max_cycles=est_max_cycles, thresh_ci=est_thresh)

        delta_e_i = ()
        if all( (delta_elec,init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec)
        end

        
        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = FermiCG.possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci) 


        #for (key, value) in basis
        #    basis_i[key] = value
	#    println(key)
	#    display(value)
        #end


        for sec in sectors
            if sec in keys(basis) 
                basis_i[sec] = basis[sec]
		#display(basis[sec])
		#st = "fock_"*string(ci.idx)*"_"*string(sec)
		#npzwrite(st, Matrix(basis[sec]))
            else
	    	#println(sec)
            end
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#
    
