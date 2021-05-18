using Combinatorics
using InteractiveUtils


"""
    ops::Tuple{String}
    delta::TransferConfig{1}
    parity::Tuple{Int}
    clusters::Tuple{Cluster}
    ints::Array{Float64}
    cache::Dict


input:
- delta = list of change of Na,Nb,state
			e.g., [(-1,-1),(1,1),(0,0)] means alpha and beta transition
			from cluster 1 to 2, cluster 3 is fock diagonal
- ops   = list of operators
			e.g., ["ab","AB",""]

- ints  = tensor containing the integrals for this block
			e.g., ndarray([p,q,r,s]) where p,q are in 1 and r,s are in 2

- data contained in object
		active: list of clusters which have non-identity operators
			this includes fock-diagonal couplings,
			e.g., ["Aa","","Bb"] would have active = [0,2]
- parity: does each operator have even or odd number of second quantized operators 
"""
abstract type ClusteredTerm end

struct ClusteredTerm1B <: ClusteredTerm
    ops::Tuple{String}
    delta::TransferConfig{1}
    parity::Tuple{Int}
    clusters::Tuple{Cluster}
    ints::Array{Float64,1}
    cache::Dict
end

struct ClusteredTerm2B <: ClusteredTerm
    ops::Tuple{String,String}
    #delta::Tuple{Tuple{Int16,Int16},Tuple{Int16,Int16}}
    delta::TransferConfig{2}
    parity::Tuple{Int,Int}
    #active::Vector{Int16}
    clusters::Tuple{Cluster,Cluster}
    ints::Array{Float64,2}
    cache::Dict
end

struct ClusteredTerm3B <: ClusteredTerm
    ops::Tuple{String,String,String}
    delta::TransferConfig{3}
    parity::Tuple{Int,Int,Int}
    #active::Vector{Int16}
    clusters::Tuple{Cluster,Cluster,Cluster}
    ints::Array{Float64,3}
    cache::Dict
end

struct ClusteredTerm4B <: ClusteredTerm
    ops::Tuple{String,String,String,String}
    delta::TransferConfig{4}
    parity::Tuple{Int,Int,Int,Int}
    clusters::Tuple{Cluster,Cluster,Cluster,Cluster}
    ints::Array{Float64,4}
    cache::Dict
end

#function ClusteredTerm(ops, delta::Vector{Tuple{Int}}, clusters, ints)
#end

function Base.display(t::ClusteredTerm1B)
    @printf( " 1B: %2i          :", t.clusters[1].idx)
    println(t.ops)
end
function Base.display(t::ClusteredTerm2B)
    @printf( " 2B: %2i %2i       :", t.clusters[1].idx, t.clusters[2].idx)
    println(t.ops)
end
function Base.display(t::ClusteredTerm3B)
    @printf( " 3B: %2i %2i %2i    :", t.clusters[1].idx, t.clusters[2].idx, t.clusters[3].idx)
    println(t.ops)
end
function Base.display(t::ClusteredTerm4B)
    @printf( " 4B: %2i %2i %2i %2i :", t.clusters[1].idx, t.clusters[2].idx, t.clusters[3].idx, t.clusters[4].idx)
    println(t.ops)
end

   
#######################################################################################################333
"""    
    trans::Dict{TransferConfig,Vector{ClusteredTerm}}
"""
struct ClusteredOperator
    trans::Dict{TransferConfig,Vector{ClusteredTerm}}
end
ClusteredOperator() = ClusteredOperator(Dict{TransferConfig,Vector{ClusteredTerm}}())
function flush_cache(clustered_ham::ClusteredOperator)
    for (ftrans, terms) in clustered_ham.trans
        for term in terms
            for cached in keys(term.cache)
                delete!(term.cache, cached)
            end
            #display(term.cache)
        end
    end
end
Base.haskey(co::ClusteredOperator, i) = return haskey(co.trans,i)
Base.setindex!(co::ClusteredOperator, i, j) = return co.trans[j] = i
Base.getindex(co::ClusteredOperator, i) = return co.trans[i]
Base.iterate(co::ClusteredOperator, state=1) = iterate(co.trans, state)
flush_cache(h::Dict{TransferConfig,Vector{ClusteredTerm}}) = flush_cache(ClusteredOperator(h)) 
mem_used_by_cache(h::Dict{TransferConfig,Vector{ClusteredTerm}}) = mem_used_by_cache(ClusteredOperator(h)) 
Base.convert(::Type{ClusteredOperator}, input::Dict{TransferConfig,Vector{ClusteredTerm}}) = ClusteredOperator(input)

function Base.display(co::ClusteredOperator)
    for (ftrans, terms) in co
        display(ftrans)
        for term in terms
            display(term)
        end
    end
end

function mem_used_by_cache(h::ClusteredOperator)
    mem = 0.0
    for (ftrans, terms) in h.trans
        for term in terms
            mem += sizeof(term.cache)
        end
    end
    return mem
end

function bubble_sort(inp)
    #={{{=#
    cmpcount, swapcount = 0, 0
    blist = copy(inp)
    bperm = collect(1:length(inp))
    for j in 1:length(blist)
        for i in 1:(length(blist)-j)
            cmpcount += 1
            if blist[i] > blist[i+1]
                swapcount += 1
                blist[i], blist[i+1] = blist[i+1], blist[i]
                bperm[i], bperm[i+1] = bperm[i+1], bperm[i]
            end
        end
    end
    return bperm, swapcount
#=}}}=#
end


"""
    extract_terms(ints::InCoreInts, clusters)

Extract all ClusteredTerm types from a given 1e integral tensor 
and a list of clusters
returns `terms::Dict{TransferConfig,Vector{ClusteredTerm}}`
"""
function extract_ClusteredTerms(ints::InCoreInts, clusters)
    norb = 0
    for ci in clusters
        norb += length(ci)
    end
    length(size(ints.h1)) == 2 || throw(Exception)
    size(ints.h1,1) == norb || throw(Exception)
    size(ints.h1,2) == norb || throw(Exception)

    #terms = Dict{TransferConfig,Vector{ClusteredTerm}}()
    terms = ClusteredOperator() 
    #terms = Dict{Vector{Tuple{Int16,Int16}},Vector{ClusteredTerm}}()
    #terms = Dict{Tuple,Vector{ClusteredTerm}}()
    n_clusters = length(clusters)
    ops_a = Array{String}(undef,n_clusters)
    ops_b = Array{String}(undef,n_clusters)
    fill!(ops_a,"")
    fill!(ops_b,"")
  
    zero_fock = TransferConfig([(0,0) for i in clusters])
    #zero_fock::Vector{Tuple{Int16,Int16}} = [(0,0) for i in clusters]
    #zero_fock = Tuple([(0,0) for i in clusters])
    terms[zero_fock] = Vector{ClusteredTerm}()
   
    # 1-body terms
    if true 
        for ci in clusters
#={{{=#
            # instead of forming p'q and p'q'sr just precontract and keep them in 
            # ClusterOps
            term = ClusteredTerm1B(("H",), ((0,0),), (0,), (ci,), zeros(1),Dict())
            push!(terms[zero_fock],term)
#=}}}=#
        end
    end

    # 2-body 1-electron terms
    if true 
        for ci in clusters
            for cj in clusters
                #={{{=#
                i = ci.idx
                j = cj.idx

                i < j || continue

                spin_cases =[["A","a"],
                             ["B","b"]
                            ]

                fock_cases =[[(1,0),(-1,0)],
                             [(0,1),(0,-1)]
                            ]

                termstr = []
                append!(termstr,unique(permutations([ci,cj])))
                #append!(termstr,unique(permutations([ci,cj,cj,cj])))
                #append!(termstr,unique(permutations([ci,ci,ci,cj])))

                #
                #   (pr|qs) p'q'sr
                #
                for term in termstr 

                    #
                    #   find permutations and sign needed to sort the indices 
                    #   such that clusters increase from left to right
                    perm, countswap = bubble_sort(term) 
                    perm == sortperm(term, alg=MergeSort)|| throw(Exception) 

                    permsign = 1
                    if countswap%2 != 0 
                        permsign = -1
                    end

                    #vprqs = view(ints.h2,term[1].orb_list, term[2].orb_list, term[3].orb_list, term[4].orb_list) 
                    hpq = view(ints.h1,term[1].orb_list, term[2].orb_list) 

                    #
                    # now align (pqsr) ints so that they align with indices from operators after sorting
                    # in this ordering, one can simply contract by sum(v .* d)
                    h = permsign .* permutedims(hpq,perm)


                    for sidx in 1:length(spin_cases)
                        oper = spin_cases[sidx][perm]
                        fock = fock_cases[sidx][perm]
                        oper1 = ""
                        oper2 = ""
                        fock1 = [0,0]
                        fock2 = [0,0]
                        for cidx in 1:length(term[perm])
                            if term[perm][cidx] == ci
                                oper1 *= oper[cidx]
                                fock1 .+= fock[cidx]
                            elseif term[perm][cidx] == cj
                                oper2 *= oper[cidx]
                                fock2 .+= fock[cidx]
                            else
                                throw(Exception)
                            end
                        end

                        parity1 = 0
                        parity2 = 0
                        if length(oper1)%2 != 0
                            parity1 = 1
                        end
                        if length(oper2)%2 != 0
                            parity2 = 1
                        end
                        parity = (parity1, parity2)

                        clusteredterm = ClusteredTerm2B((oper1,oper2), (Tuple(fock1),Tuple(fock2)), parity, (ci, cj), h, Dict())
                        #display(clusteredterm)
                        focktrans = replace(zero_fock, (ci.idx, cj.idx), (fock1, fock2))
#                        focktrans = [zero_fock...]
#                        focktrans[ci.idx] = Tuple(fock1)
#                        focktrans[cj.idx] = Tuple(fock2)
#                        focktrans = TransferConfig(focktrans)
                        if haskey(terms,focktrans)
                            push!(terms[focktrans], clusteredterm)
                        else
                            terms[focktrans] = [clusteredterm]
                        end

                    end
                end
            end
        end
        #=}}}=#
    end
    
    # 2-body 2-electron terms
    if true 
        for ci in clusters
            for cj in clusters
                #={{{=#
                i = ci.idx
                j = cj.idx

                i < j || continue

                spin_cases =[["A","A","a","a"],
                             ["B","B","b","b"],
                             ["A","B","b","a"],
                             ["B","A","a","b"]
                            ]

                fock_cases =[[(1,0),(1,0),(-1,0),(-1,0)],
                             [(0,1),(0,1),(0,-1),(0,-1)],
                             [(1,0),(0,1),(0,-1),(-1,0)],
                             [(0,1),(1,0),(-1,0),(0,-1)]
                            ]


                termstr = []
                append!(termstr,unique(permutations([ci,ci,cj,cj])))
                append!(termstr,unique(permutations([ci,cj,cj,cj])))
                append!(termstr,unique(permutations([ci,ci,ci,cj])))

                #
                #   (pr|qs) p'q'sr
                #
                for term in termstr 

                    #
                    #   find permutations and sign needed to sort the indices 
                    #   such that clusters increase from left to right
                    perm, countswap = bubble_sort(term) 
                    perm == sortperm(term, alg=MergeSort)|| error("problem with bubble_sort") 
            
                    permsign = 1
                    if countswap%2 != 0 
                        permsign = -1
                    end

                    vprqs = view(ints.h2,term[1].orb_list, term[4].orb_list, term[2].orb_list, term[3].orb_list) 
                    #
                    # align (prqs) ints so that they align with indices from operators before sorting
                    vpqsr = permutedims(vprqs,[1,3,4,2])

                    #
                    # now align (pqsr) ints so that they align with indices from operators after sorting
                    # in this ordering, one can simply contract by sum(v .* d)
                    v = (.5 * permsign) .* permutedims(vpqsr,perm)

                    #
                    # now reshape ints so they can be contracted like gamma(pqr) V(pqr,s) gamma(s)
                    newshape = [1,1]

                    for cidx in 1:length(term[perm])
                        if term[perm][cidx] == ci
                            newshape[1] *= size(v,cidx)
                        elseif term[perm][cidx] == cj
                            newshape[2] *= size(v,cidx)
                        else
                            throw(Exception)
                        end
                    end

                    for sidx in 1:length(spin_cases)
                        oper = spin_cases[sidx][perm]
                        fock = fock_cases[sidx][perm]
                        oper1 = ""
                        oper2 = ""
                        fock1 = [0,0]
                        fock2 = [0,0]
                        for cidx in 1:length(term[perm])
                            if term[perm][cidx] == ci
                                oper1 *= oper[cidx]
                                fock1 .+= fock[cidx]
                            elseif term[perm][cidx] == cj
                                oper2 *= oper[cidx]
                                fock2 .+= fock[cidx]
                            else
                                throw(Exception)
                            end
                        end
                        vcurr = deepcopy(v)
                       
                        if true 
                            if oper1 == "BA"
                                oper1 = "AB"
                                vcurr = -permutedims(vcurr,[2,1,3,4])
                            elseif oper1 == "ba"
                                oper1 = "ab"
                                vcurr = -permutedims(vcurr,[2,1,3,4])
                            elseif oper1 == "BAa"
                                oper1 = "ABa"
                                vcurr = -permutedims(vcurr,[2,1,3,4])
                            elseif oper1 == "BAb"
                                oper1 = "ABb"
                                vcurr = -permutedims(vcurr,[2,1,3,4])
                            elseif oper1 == "Bab"
                                oper1 = "Bba"
                                vcurr = -permutedims(vcurr,[1,3,2,4])
                            elseif oper1 == "Aab"
                                oper1 = "Aba"
                                vcurr = -permutedims(vcurr,[1,3,2,4])
                            end
                            

                            if oper2 == "BA"
                                oper2 = "AB"
                                vcurr = -permutedims(vcurr,[1,2,4,3])
                            elseif oper2 == "ba"
                                oper2 = "ab"
                                vcurr = -permutedims(vcurr,[1,2,4,3])
                            elseif oper2 == "BAa"
                                oper2 = "ABa"
                                vcurr = -permutedims(vcurr,[1,3,2,4])
                            elseif oper2 == "BAb"
                                oper2 = "ABb"
                                vcurr = -permutedims(vcurr,[1,3,2,4])
                            elseif oper2 == "Bab"
                                oper2 = "Bba"
                                vcurr = -permutedims(vcurr,[1,2,4,3])
                            elseif oper2 == "Aab"
                                oper2 = "Aba"
                                vcurr = -permutedims(vcurr,[1,2,4,3])
                            end
                        end
                       
                        vcurr = copy(reshape(vcurr,newshape...))

                        parity1 = 0
                        parity2 = 0
                        if length(oper1)%2 != 0
                            parity1 = 1
                        end
                        if length(oper2)%2 != 0
                            parity2 = 1
                        end
                        parity = (parity1, parity2)
                        
                        clusteredterm = ClusteredTerm2B((oper1,oper2), (Tuple(fock1),Tuple(fock2)), parity, (ci, cj), vcurr, Dict())
                        #display(clusteredterm)
                        focktrans = replace(zero_fock, (ci.idx, cj.idx), (fock1, fock2))
                        if haskey(terms,focktrans)
                            push!(terms[focktrans], clusteredterm)
                        else
                            terms[focktrans] = [clusteredterm]
                        end

                    end

                    #all(isapprox(vpqsr,0.0)) || continue
                end

                #=}}}=#
            end
        end
    end

    # 3-body 2-electron terms
    if true 
        for ci in clusters
            for cj in clusters
                for ck in clusters
                    #={{{=#
                    i = ci.idx
                    j = cj.idx
                    k = ck.idx

                    i < j < k || continue

                    spin_cases =[["A","A","a","a"],
                                 ["B","B","b","b"],
                                 ["A","B","b","a"],
                                 ["B","A","a","b"]
                                ]

                    fock_cases =[[(1,0),(1,0),(-1,0),(-1,0)],
                                 [(0,1),(0,1),(0,-1),(0,-1)],
                                 [(1,0),(0,1),(0,-1),(-1,0)],
                                 [(0,1),(1,0),(-1,0),(0,-1)]
                                ]


                    termstr = []
                    append!(termstr,unique(permutations([ci,ci,cj,ck])))
                    append!(termstr,unique(permutations([ci,cj,cj,ck])))
                    append!(termstr,unique(permutations([ci,cj,ck,ck])))

                    #
                    #   (pr|qs) p'q'sr
                    #
                    for term in termstr 

                        #
                        #   find permutations and sign needed to sort the indices 
                        #   such that clusters increase from left to right
                        perm, countswap = bubble_sort(term) 
                        perm == sortperm(term, alg=MergeSort)|| error("problem with bubble_sort") 

                        permsign = 1
                        if countswap%2 != 0 
                            permsign = -1
                        end

                        vprqs = view(ints.h2,term[1].orb_list, term[4].orb_list, term[2].orb_list, term[3].orb_list) 
                        #
                        # align (prqs) ints so that they align with indices from operators before sorting
                        vpqsr = permutedims(vprqs,[1,3,4,2])

                        #
                        # now align (pqsr) ints so that they align with indices from operators after sorting
                        # in this ordering, one can simply contract by sum(v .* d)
                        v = (.5 * permsign) .* permutedims(vpqsr,perm)

                        #
                        # now reshape ints so they can be contracted like gamma(pqr) V(pqr,s) gamma(s)
                        newshape = [1,1,1]

                        for cidx in 1:length(term[perm])
                            if term[perm][cidx] == ci
                                newshape[1] *= size(v,cidx)
                            elseif term[perm][cidx] == cj
                                newshape[2] *= size(v,cidx)
                            elseif term[perm][cidx] == ck
                                newshape[3] *= size(v,cidx)
                            else
                                throw(Exception)
                            end
                        end

                        for sidx in 1:length(spin_cases)
                            oper = spin_cases[sidx][perm]
                            fock = fock_cases[sidx][perm]
                            oper1 = ""
                            oper2 = ""
                            oper3 = ""
                            fock1 = [0,0]
                            fock2 = [0,0]
                            fock3 = [0,0]
                            for cidx in 1:length(term[perm])
                                if term[perm][cidx] == ci
                                    oper1 *= oper[cidx]
                                    fock1 .+= fock[cidx]
                                elseif term[perm][cidx] == cj
                                    oper2 *= oper[cidx]
                                    fock2 .+= fock[cidx]
                                elseif term[perm][cidx] == ck
                                    oper3 *= oper[cidx]
                                    fock3 .+= fock[cidx]
                                else
                                    throw(Exception)
                                end
                            end
                            vcurr = deepcopy(v)

                            if true 
                                if oper1 == "BA"
                                    oper1 = "AB"
                                    vcurr = -permutedims(vcurr,[2,1,3,4])
                                elseif oper1 == "ba"
                                    oper1 = "ab"
                                    vcurr = -permutedims(vcurr,[2,1,3,4])
                                end


                                if oper2 == "BA"
                                    oper2 = "AB"
                                    vcurr = -permutedims(vcurr,[1,3,2,4])
                                elseif oper2 == "ba"
                                    oper2 = "ab"
                                    vcurr = -permutedims(vcurr,[1,3,2,4])
                                end


                                if oper3 == "BA"
                                    oper3 = "AB"
                                    vcurr = -permutedims(vcurr,[1,2,4,3])
                                elseif oper3 == "ba"
                                    oper3 = "ab"
                                    vcurr = -permutedims(vcurr,[1,2,4,3])
                                end
                            end

                            vcurr = copy(reshape(vcurr,newshape...))

                            #core,factors = tucker_decompose(vcurr)
                            
                            parity1 = 0
                            parity2 = 0
                            parity3 = 0
                            if length(oper1)%2 != 0
                                parity1 = 1
                            end
                            if length(oper2)%2 != 0
                                parity2 = 1
                            end
                            if length(oper3)%2 != 0
                                parity3 = 1
                            end
                            parity = (parity1, parity2, parity3)

                            clusteredterm = ClusteredTerm3B((oper1,oper2,oper3), (Tuple(fock1),Tuple(fock2),Tuple(fock3)), parity, (ci, cj, ck), vcurr, Dict())
                            #display(clusteredterm)
                            focktrans = replace(zero_fock, (ci.idx, cj.idx, ck.idx), (fock1, fock2, fock3))
                            if haskey(terms,focktrans)
                                push!(terms[focktrans], clusteredterm)
                            else
                                terms[focktrans] = [clusteredterm]
                            end
                        end
                        #all(isapprox(vpqsr,0.0)) || continue
                    end

                    #=}}}=#
                end
            end
        end
    end

    # 4-body 2-electron terms
    if true 
        for ci in clusters
            for cj in clusters
                for ck in clusters
                    for cl in clusters
                        #={{{=#
                        i = ci.idx
                        j = cj.idx
                        k = ck.idx
                        l = cl.idx

                        i < j < k < l|| continue

                        spin_cases =[["A","A","a","a"],
                                     ["B","B","b","b"],
                                     ["A","B","b","a"],
                                     ["B","A","a","b"]
                                    ]

                        fock_cases =[[(1,0),(1,0),(-1,0),(-1,0)],
                                     [(0,1),(0,1),(0,-1),(0,-1)],
                                     [(1,0),(0,1),(0,-1),(-1,0)],
                                     [(0,1),(1,0),(-1,0),(0,-1)]
                                    ]


                        termstr = []
                        append!(termstr,unique(permutations([ci,cj,ck,cl])))

                        #
                        #   (pr|qs) p'q'sr
                        #
                        for term in termstr 

                            #
                            #   find permutations and sign needed to sort the indices 
                            #   such that clusters increase from left to right
                            perm, countswap = bubble_sort(term) 
                            perm == sortperm(term, alg=MergeSort)|| error("problem with bubble_sort") 

                            permsign = 1
                            if countswap%2 != 0 
                                permsign = -1
                            end

                            vprqs = view(ints.h2,term[1].orb_list, term[4].orb_list, term[2].orb_list, term[3].orb_list) 
                            #
                            # align (prqs) ints so that they align with indices from operators before sorting
                            vpqsr = permutedims(vprqs,[1,3,4,2])

                            #
                            # now align (pqsr) ints so that they align with indices from operators after sorting
                            # in this ordering, one can simply contract by sum(v .* d)
                            v = (.5 * permsign) .* permutedims(vpqsr,perm)

                            #
                            # no reshape needed 

                            for sidx in 1:length(spin_cases)
                                oper = spin_cases[sidx][perm]
                                fock = fock_cases[sidx][perm]
                                oper1 = ""
                                oper2 = ""
                                oper3 = ""
                                oper4 = ""
                                fock1 = [0,0]
                                fock2 = [0,0]
                                fock3 = [0,0]
                                fock4 = [0,0]
                                for cidx in 1:length(term[perm])
                                    if term[perm][cidx] == ci
                                        oper1 *= oper[cidx]
                                        fock1 .+= fock[cidx]
                                    elseif term[perm][cidx] == cj
                                        oper2 *= oper[cidx]
                                        fock2 .+= fock[cidx]
                                    elseif term[perm][cidx] == ck
                                        oper3 *= oper[cidx]
                                        fock3 .+= fock[cidx]
                                    elseif term[perm][cidx] == cl
                                        oper4 *= oper[cidx]
                                        fock4 .+= fock[cidx]
                                    else
                                        throw(Exception)
                                    end
                                end

                            
                                parity1 = 0
                                parity2 = 0
                                parity3 = 0
                                parity4 = 0
                                if length(oper1)%2 != 0
                                    parity1 = 1
                                end
                                if length(oper2)%2 != 0
                                    parity2 = 1
                                end
                                if length(oper3)%2 != 0
                                    parity3 = 1
                                end
                                if length(oper4)%2 != 0
                                    parity4 = 1
                                end
                                parity = (parity1, parity2, parity3, parity4)
                                
                                clusteredterm = ClusteredTerm4B((oper1,oper2,oper3,oper4), (Tuple(fock1),Tuple(fock2),Tuple(fock3),Tuple(fock4)), parity, (ci, cj, ck, cl), v, Dict())
                                focktrans = replace(zero_fock, (ci.idx, cj.idx, ck.idx, cl.idx), (fock1, fock2, fock3, fock4))
                                if haskey(terms,focktrans)
                                    push!(terms[focktrans], clusteredterm)
                                else
                                    terms[focktrans] = [clusteredterm]
                                end
                            end
                        end
                        #=}}}=#
                    end
                end
            end
        end
    end

    unique!(terms)
    
    return terms
end


"""
    extract_S2(clusters)

Form a clustered operator type for the S^2 operator
"""
function extract_S2(clusters)
            #={{{=#

    norb = 0
    for ci in clusters
        norb += length(ci)
    end

    terms = ClusteredOperator() 
    n_clusters = length(clusters)
    ops_a = Array{String}(undef,n_clusters)
    ops_b = Array{String}(undef,n_clusters)
    fill!(ops_a,"")
    fill!(ops_b,"")
  
    zero_fock = TransferConfig([(0,0) for i in clusters])
    terms[zero_fock] = Vector{ClusteredTerm}()
    for ci in clusters
        fock1 = (0,0)
        clusteredterm = ClusteredTerm1B(("S2",), (fock1,), (0,), (ci, ), ones(1), Dict())

        focktrans = replace(zero_fock, (ci.idx,), (fock1,))
        if haskey(terms,focktrans)
            push!(terms[focktrans], clusteredterm)
        else
            terms[focktrans] = [clusteredterm]
        end
        
        for cj in clusters
            i = ci.idx
            j = cj.idx

            i < j || continue


            fock1 = (1,-1)
            fock2 = (-1,1)
            clusteredterm = ClusteredTerm2B(("S+","S-"), (fock1,fock2), (0,0), (ci, cj), ones(length(ci),length(cj)), Dict())

            focktrans = replace(zero_fock, (ci.idx, cj.idx), (fock1, fock2))
            if haskey(terms,focktrans)
                push!(terms[focktrans], clusteredterm)
            else
                terms[focktrans] = [clusteredterm]
            end
            
            fock1 = (-1,1)
            fock2 = (1,-1)
            clusteredterm = ClusteredTerm2B(("S-","S+"), (fock1,fock2), (0,0), (ci, cj), ones(length(ci),length(cj)), Dict())

            focktrans = replace(zero_fock, (ci.idx, cj.idx), (fock1, fock2))
            if haskey(terms,focktrans)
                push!(terms[focktrans], clusteredterm)
            else
                terms[focktrans] = [clusteredterm]
            end
            
            fock1 = (0,0)
            fock2 = (0,0)
            clusteredterm = ClusteredTerm2B(("Sz","Sz"), (fock1,fock2), (0,0), (ci, cj), 2*ones(1,1), Dict())

            focktrans = replace(zero_fock, (ci.idx, cj.idx), (fock1, fock2))
            if haskey(terms,focktrans)
                push!(terms[focktrans], clusteredterm)
            else
                terms[focktrans] = [clusteredterm]
            end
        end
    end
    return terms
end
#=}}}=#


"""
    unique!(clustered_ham::ClusteredOperator)

combine terms to keep only unique operators
"""
function unique!(clustered_ham::ClusteredOperator)
#function unique!(clustered_ham::Dict{TransferConfig,Vector{ClusteredTerm}})
#={{{=#
    println(" Remove duplicates")
    #
    # first just remove duplicates
    nstart = 0
    nfinal = 0
#    for (ftrans, terms) in clustered_ham
#        tmp = deepcopy(terms)
#        for term in terms
#            println(term.ops)
##            swap = false
##            idx = 0
##            for (opidx,op) in enumerate(term.ops)
##                if op == "BA"
##                    swap = true
##                    idx = opidx
##                end
##            end
##            if swap
##                println(size(term.ints))
##                v = term.ints 
##                v = copy(reshape(v, (size(v,1), size(v,2), size(v,3), size(v,4))))
##                v = -permutedims(v,[2,1,3,4])
##                v = reshape(v, (size(v,1)*size(v,2)*size(v,3), size(v,4)))
##
##                newterm = ClusteredTerm2B(("ABb","a"), term.delta, term.clusters, term.ints)
##                push!(tmp,newterm)
##            end
#
#            if term.ops == ("BAb","a")
#               
#                println(size(term.ints))
#                v = term.ints 
#                v = copy(reshape(v, (size(v,1), size(v,2), size(v,3), size(v,4))))
#                v = -permutedims(v,[2,1,3,4])
#                v = reshape(v, (size(v,1)*size(v,2)*size(v,3), size(v,4)))
#
#                newterm = ClusteredTerm2B(("ABb","a"), term.delta, term.clusters, term.ints)
#                push!(tmp,newterm)
#            else
#                push!(tmp,term)
#            end
#        end
#
#        clustered_ham[ftrans] = tmp
#    end
        
#        tmp = deepcopy(terms)
#        for term in tmp
#            for op in term.ops
#                if op == "BAb"
#
#                    fconfig = 
#                    newterm = ClusteredTerm2B((oper1,oper2,oper3,oper4), ftrans, term.clusters, v)



    for (ftrans, terms) in clustered_ham
        unique = Dict()
        for term in terms
            nstart += 1
            keystr = ""
            for (i,j) in zip(term.ops,term.clusters)
                keystr *= string(i,"(",j.idx,")")
            end
            if haskey(unique,keystr)
                unique[keystr].ints .+= term.ints
            else
                unique[keystr] = deepcopy(term)
            end
        end
        clustered_ham[ftrans] = Vector{ClusteredTerm}()
        for (keystr, term) in unique
            push!(clustered_ham[ftrans],term)
        end
        nfinal += length(clustered_ham[ftrans]) 
    end

    @printf(" Number of terms reduced from %5i to %5i\n", nstart, nfinal)
#=}}}=#
end


function check_term(term::ClusteredTerm1B, 
                            fock_bra::FockConfig, bra::T, 
                            fock_ket::FockConfig, ket::T) where T<:Union{ClusterConfig, TuckerConfig}
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    fock_bra[term.clusters[1].idx] == fock_ket[term.clusters[1].idx] .+ term.delta[1] || return false 
    return true
end

function check_term(term::ClusteredTerm2B, 
                            fock_bra::FockConfig, bra::T, 
                            fock_ket::FockConfig, ket::T) where T<:Union{ClusterConfig, TuckerConfig}
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    #
    #
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    fock_bra[term.clusters[1].idx] == fock_ket[term.clusters[1].idx] .+ term.delta[1] || return false 
    fock_bra[term.clusters[2].idx] == fock_ket[term.clusters[2].idx] .+ term.delta[2] || return false 
    return true
end

function check_term(term::ClusteredTerm3B, 
                            fock_bra::FockConfig, bra::T, 
                            fock_ket::FockConfig, ket::T) where T<:Union{ClusterConfig, TuckerConfig}
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue
        ci != term.clusters[3].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    fock_bra[term.clusters[1].idx] == fock_ket[term.clusters[1].idx] .+ term.delta[1] || return false 
    fock_bra[term.clusters[2].idx] == fock_ket[term.clusters[2].idx] .+ term.delta[2] || return false 
    fock_bra[term.clusters[3].idx] == fock_ket[term.clusters[3].idx] .+ term.delta[3] || return false 
    return true
end

function check_term(term::ClusteredTerm4B, 
                            fock_bra::FockConfig, bra::T, 
                            fock_ket::FockConfig, ket::T) where T<:Union{ClusterConfig, TuckerConfig}
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue
        ci != term.clusters[3].idx || continue
        ci != term.clusters[4].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    fock_bra[term.clusters[1].idx] == fock_ket[term.clusters[1].idx] .+ term.delta[1] || return false 
    fock_bra[term.clusters[2].idx] == fock_ket[term.clusters[2].idx] .+ term.delta[2] || return false 
    fock_bra[term.clusters[3].idx] == fock_ket[term.clusters[3].idx] .+ term.delta[3] || return false 
    fock_bra[term.clusters[4].idx] == fock_ket[term.clusters[4].idx] .+ term.delta[4] || return false 
    return true
end




function compute_terms_state_sign(term::ClusteredTerm, fock_ket::FockConfig)
    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = 1
    for (oi,o) in enumerate(term.ops)
        if term.parity[oi] == 1  #only count electrons if operator is odd
            n_elec_hopped = 0
            for ci in 1:term.clusters[oi].idx-1
                n_elec_hopped += fock_ket[ci][1] + fock_ket[ci][2]
            end
            if n_elec_hopped % 2 != 0
                state_sign = -state_sign
            end
        end
    end
    return state_sign
end


function print_fock_sectors(sector::Vector{Tuple{T,T}}) where T<:Integer
    print("  ")
    for ci in sector
        @printf("(%iα,%iβ)", ci[1],ci[2])
    end
    println()
end




"""
    extract_1body_operator(clustered_ham::ClusteredOperator; op_string="H")

Extract a 1-body operator for use in perturbation theory
- `op_string`: either H or Hcmf
"""
function extract_1body_operator(clustered_ham::ClusteredOperator; op_string="H")
    out = ClusteredOperator()
    for (ftrans, terms) in clustered_ham
        for term in terms
            if term isa ClusteredTerm1B
                
                term2 = ClusteredTerm1B((op_string,), term.delta, term.parity, term.clusters, term.ints, term.cache)
                if haskey(out, ftrans)
                    push!(out[ftrans], term2)
                else
                    out[ftrans] = [term2]
                end
            end
        end
    end
    return out
end

