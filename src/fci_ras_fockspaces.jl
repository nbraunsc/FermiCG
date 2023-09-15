using FermiCG

function add_fock_configs_for_rasci(ref_fock::FockConfig, ci_vector::TPSCIstate; n_clusters=6, ex_level=1)

    A = Vector{Tuple{Tuple{Int16, Int16}, Tuple{Int16, Int16}, Tuple{Int16, Int16}}}()
    if ex_level == 1
        push!(A, (( 0, 0), (0, 0), (0, 0)),
              ((-1, 0), (1, 0), (0, 0)),
              (( 0,-1), (0, 1), (0, 0)),
              (( 0, 0), (-1, 0), (1, 0)),
              (( 0, 0), (0, -1), (0, 1)),
              ((-1, 0), (0, 0), (1, 0)),
              ((0, -1), (0, 0), (0, 1)),
              ((-1, 0), (1, -1), (0, 1)),
              (( 0, -1), (-1, 1), (1, 0)))

        B = A
        for one in A
            for two in B
                tmp_fspace = FermiCG.replace(ref_fock, collect(1:n_clusters), (one[1].+ref_fock[1], one[2].+ref_fock[2], one[3].+ref_fock[3], two[1].+ref_fock[4], two[2].+ref_fock[5], two[3].+ref_fock[6]))
                FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
            end
        end
    end
end



