using LinearAlgebra
using FermiCG
using Printf
using Test
using LinearMaps
using Arpack

using Profile 

atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[0,0,1]))
push!(atoms,Atom(3,"H",[0,0,2]))
push!(atoms,Atom(4,"H",[0,0,3]))
push!(atoms,Atom(5,"H",[0,0,4]))
push!(atoms,Atom(6,"H",[0,0,5]))
push!(atoms,Atom(7,"H",[0,0,6]))
push!(atoms,Atom(8,"H",[0,0,7]))
#push!(atoms,Atom(9,"H",[0,0,8]))
#push!(atoms,Atom(10,"H",[0,0,9]))
#push!(atoms,Atom(11,"H",[0,0,10]))
#push!(atoms,Atom(12,"H",[0,0,11]))
#basis = "6-31g"
basis = "sto-3g"

mol     = Molecule(0,1,atoms)
mf = FermiCG.pyscf_do_scf(mol,basis)
ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);
e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,4,4)
# @printf(" FCI Energy: %12.8f\n", e_fci)


norbs = size(ints.h1)[1]

problem = FCIProblem(norbs, 4, 4)

display(problem)

#@time Hmat = FermiCG.build_H_matrix(ints, problem)
#@time e,v = eigs(Hmat, nev = 10, which=:SR)
#e = real(e)
#for ei in e
#    @printf(" Energy: %12.8f\n",ei+ints.h0)
#end


print(" Compute spin_diagonal terms\n")
@time Hdiag_a = FermiCG.precompute_spin_diag_terms(ints,problem,problem.na)
@time Hdiag_b = FermiCG.precompute_spin_diag_terms(ints,problem,problem.nb)
print(" done\n")

Hmap = FermiCG.get_map(ints, problem, Hdiag_a, Hdiag_b)

v = zeros(problem.dim,1)
v[1] = 1

#Hmat = .5*(Hmat + transpose(Hmat))
@time e,v = eigs(Hmap, v0=v[:,1], nev = 1, which=:SR, tol=1e-6, maxiter=12)
e = real(e)
for ei in e
    @printf(" Energy: %12.8f\n",ei+ints.h0)
end
 
