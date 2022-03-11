using FermiCG
using Printf
using Test
using JLD2 

#@testset "BSTstate" begin
#if true 
    @load "_testdata_cmf_h6.jld2"
    v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
    
    e_ci, v_ci = FermiCG.tucker_ci_solve(v, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.31710895, atol=1e-8)

   
    v = FermiCG.BSTstate(v,R=1)
    xspace  = FermiCG.build_compressed_1st_order_state(v, cluster_ops, clustered_ham, nbody=4, thresh=1e-4)
    xspace = FermiCG.compress(xspace, thresh=1e-3)
    display(size(xspace))

    FermiCG.nonorth_add!(v, xspace)
    v = FermiCG.BSTstate(v,R=20)
    FermiCG.randomize!(v)
    FermiCG.orthonormalize!(v)
    
    e_ci, v = FermiCG.tucker_ci_solve(v, cluster_ops, clustered_ham, 
                                      conv_thresh = 1e-7,
                                      max_iter    = 100,
                                      max_ss_vecs = 30,
                                     )
    
    #e_pt, v_pt = FermiCG.do_fois_pt2(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    
    if false 
        e_var, v_var = FermiCG.block_sparse_tucker(v, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               max_iter_pt = 200, 
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-3,
                                               thresh_pt   = 1e-3,
                                               ci_conv     = 1e-5,
                                               ci_max_iter = 100,
                                               ci_max_ss_vecs = 100,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4)
    end
#end

if false
@testset "BST" begin


    @load "_testdata_cmf_h6.jld2"

    v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)

    e_var, v_var = FermiCG.block_sparse_tucker(v, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               max_iter_pt = 200, 
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-3,
                                               thresh_pt   = sqrt(1e-5),
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = true,
                                               tol_tucker  = 1e-4)

    @test isapprox(e_var[1], -18.329454973117784, atol=1e-8)
    

    e_cepa, v_cepa = FermiCG.do_fois_cepa(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_cepa)
    @test isapprox(e_cepa[1], -18.32978899988935, atol=1e-8)
    
    e_pt, v_pt = FermiCG.do_fois_pt2(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_pt)
    @test isapprox(e_pt[1], -18.326970022448485, atol=1e-8)

    e_ci, v_ci = FermiCG.tucker_ci_solve(v_cepa, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.32964089848682, atol=1e-8)

end
end
