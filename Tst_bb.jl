module Tst_bb
using DelimitedFiles, FileIO, Images, OffsetArrays
include("SC_bb.jl")


image = (load("./images/seam_carving.jpg"))
aux = SC_bb.initialize_aux(image');
SC_bb.seam_carve(aux,1)
SC_bb.upd_energy(aux)
SC_bb.upd_seam(aux)
##First let us pass all the assertion checks.
intensity_gradient_bk = readdlm("./images/intensity_gradient_bk.txt")
@assert all(aux.intensity_gradient .- intensity_gradient_bk .< 1e-5)

seam_bk = readdlm("./images/seam_bk.txt")
@assert all(aux.seam .- seam_bk .== 0)

moves_bk = readdlm("./images/moves_bk.txt")
@assert all(moves_bk .- aux.moves .== 0)

aux_energy_bk = readdlm("./images/aux_energy_bk.txt")
## I have to take out the values where i have inf.
@assert all(aux.aux_energy[1:end-2,:] .- aux_energy_bk[2:end-2,:] .< 1e-5)


## Now re-initialize and let us time our code on 20 iterations.
##LETS US TIME IT NOW
#aux = SC_bb.initialize_aux(image');
#final_result = SC_bb.seam_carve(aux,20)
#println("final_result:",final_result)
#=
@info "warm up"
SC_bb.seam_carve(aux,1)
@info "go"
@time SC_bb.seam_carve(aux,20)
#=
=#
using Profile, ProfileView
aux= SC_bb.Img_Aux(image, bright, sobel_x, sobel_y, intensity_gradient, aux_energy, moves, seam, iterations);
Profile.init(delay=3)
@profview SC_bb.seam_carve(aux, 300)
=#
end ##end module

