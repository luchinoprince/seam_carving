module SC_bb

using Images, FileIO, OffsetArrays, LinearAlgebra, ExtractMacro
#=
Here i suppose I have everything transposed. Also i suppose I have the black and white borders,
 i suppose also i have the offsetted arrays, meaning that all the black/white 
contours correspondo to negative indexes in the left part. I also suppose the kernels are offsetted
to have indexes between between -1 and 1, as this makes indexing in the calculations more logically straightforward.
=#
mutable struct Img_Aux
    ##Energy part
    img:: Union{Adjoint{RGBX{Normed{UInt8,8}},Array{RGBX{Normed{UInt8,8}},2}}, Array{RGBX{Normed{UInt8,8}},2} };
    bright::OffsetArray{Float64,2,Array{Float64,2}}
    sobel_x::OffsetArray{Float64,2,Array{Float64,2}}
    sobel_y::Adjoint{Float64,OffsetArray{Float64,2,Array{Float64,2}}}
    intensity_gradient::Array{Float64, 2}
    ##Dynamic Programming Part
    aux_energy::OffsetArray{Float64,2,Array{Float64,2}}
    moves::Array{Int64,2}
    seam::Array{Int64,1}
    iterations::Int64;
end

function initialize_aux(image::Union{ Adjoint{RGBX{Normed{UInt8,8}},Array{RGBX{Normed{UInt8,8}},2}}, Array{RGBX{Normed{UInt8,8}},2} })
    """ Here i can pass ether an image, or its transpose, as usual for optimization purposes """
    rows,cols = size(image)
    cw=channelview(image)
    sobel_x = OffsetArray([[1.0,2.0,1.0] [0.0,0.0,0.0] [-1.0,-2.0,-1.0]], -1:1, -1:1)
    sobel_y = sobel_x'
    intensity_gradient = zeros((rows,cols))

    ##FOR THIS ONE I HAVE TO ADD THE BORDER OF INFINITY
    aux_energy=OffsetArray(zeros((rows+2, cols)), 0:rows+1, 1:cols)
    aux_energy[0,:] .= Inf
    aux_energy[end,:] .=  Inf
    ###################################################
    moves = zeros(Int64, (rows,cols));
    seam = zeros(Int64, cols);

    ###SINCE I USE ONLY BRIGHT, I CAN ADD THE BORDER ONLY HERE.
    bright = OffsetArray(zeros((rows+2,cols+2)), 0:rows+1, 0:cols+1)
    @inbounds for j in 1:(cols), i in 1:(rows)
        b = 0.0
        for c in 1:3
            b += cw[c, i, j]
        end
        b /= 3
        bright[i,j] = b
    end
    iterations=0
    return Img_Aux(image, bright, sobel_x, sobel_y, intensity_gradient, aux_energy, moves, seam, iterations);
end

function get_energy(aux::Img_Aux)
    """ Function that calculates the intensity gradient matrix the first time seam carving is called """
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    ## To keep avoid doing calculation in the bottom rows where I accumulate junk, I reduce the row counter by the amount
    ## of iterations already performed.
    rows, cols = size(img) .- (iterations, 0)
    @inbounds for j in 1:cols, i in 1:rows
        ## As we have added a black border, we do not need to account for special cases.
        gx, gy = 0.0, 0.0
        @inbounds for dy = -1:1, dx = -1:1
            gx += bright[i+dx, j+dy] * sobel_x[dx, dy]
            gy += bright[i+dx, j+dy] * sobel_y[dx, dy]
        end
        intensity_gradient[i,j]=sqrt(gx^2+gy^2)
    end
    return
end


function upd_energy(aux::Img_Aux)
    """ Function that updates the intensity gradient values after the first iteration, since we know that a lot of values of 
    this matrix will remain equal to the ones of the previous iteration"""
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    rows, cols= size(img) .- (iterations, 0)
    @inbounds for k in 1:cols
        row_index=seam[k]
        @inbounds for i in -2:1
            gx,gy=0.0,0.0
            if row_index+i > 0 && row_index+i <= rows
                for dx in -1:1, dy in -1:1
                    gx += bright[row_index+i+dx, k+dy] * sobel_x[dx, dy]
                    gy += bright[row_index+i+dx, k+dy] * sobel_y[dx, dy]
                end
                intensity_gradient[row_index+i, k] = sqrt(gx^2 + gy^2)
            end
        end
    end
    return
end 

function get_seam(aux::Img_Aux)
    """ Function that calculates the minimum energy seam at the first iteration trough dynamic programming"""
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    rows,cols= size(img) .- (iterations, 0) 
    @views aux_energy[1:rows,1] .= intensity_gradient[1:rows,1]
    ##We have to start from the second row
    for j in 2:cols, i in 1:rows
        ##Since i putted a white border on aux_energy, also here we do not have to account for special cases on the borders
        min, index = findmin((aux_energy[i-1,j-1], aux_energy[i,j-1], aux_energy[i+1,j-1]))
        aux_energy[i,j] = intensity_gradient[i,j]+min
        moves[i,j] = index-2
    end

    ## Now I have to create a seam
    @views bottom_index = argmin(aux_energy[1:rows,end])
    seam[end] = bottom_index
    for k in 2:cols
        seam[end+1-k] = seam[end+2-k] + moves[seam[end+2-k], end+2-k]
    end
    return
end


function upd_seam(aux::Img_Aux)
    """Function which finds the new seam by updating efficiently the aux_energy and moves matrices"""
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    rows,cols = size(img) .- (iterations, 0) 
    @views aux_energy[1:rows,1] .= intensity_gradient[1:rows,1]
    
    core_index_down = seam[1] 
    core_index_down == 1 ? core_index_up = 1 : core_index_up = seam[1]-1

    ## This two groups of "if/else" is to ensure we do not get out of the image.
    if core_index_down >= rows-1
        if core_index_down == rows
            iterator_down=0
        else
            iterator_down=1
        end
    else
        iterator_down=2
    end

    if core_index_up <= 2
        if core_index_up == 1
            iterator_up=0
        else
            iterator_up=1
        end
    else
        iterator_up=2
    end

    @inbounds for j in 2:cols
        previous_top = aux_energy[core_index_up-iterator_up, j]
        previous_down = aux_energy[core_index_down+iterator_down, j]
        incr_up = true; incr_down = true;
        @inbounds for i in (core_index_up-iterator_up:core_index_down+iterator_down)
            min, index = findmin((aux_energy[i-1,j-1], aux_energy[i,j-1], aux_energy[i+1,j-1]))
            aux_energy[i,j] = intensity_gradient[i,j]+min
            moves[i,j] = index-2
            i == core_index_up-iterator_up && (incr_up = (previous_top != aux_energy[i,j]))
            i == core_index_down+iterator_down && (incr_down = (previous_down != aux_energy[i,j]))
        end
        ## Now I have to accounte for the possibility of going eventually out of bounds, I have to account it
        ## Separately for the part going up and the part going down
        if (incr_up && core_index_up-iterator_up > 1)
            iterator_up+=1
        end
        if (incr_down && core_index_down+iterator_down < rows)
            iterator_down+=1
        end
    end

    ## Now I have to create a seam
    bottom_index = argmin(aux_energy[1:rows,end])
    seam[end] = bottom_index
    @inbounds for k in 2:cols
        seam[end+1-k] = seam[end+2-k] + moves[seam[end+2-k], end+2-k]
    end
    return
end



function remove_seam(aux::Img_Aux)
    """This function removes the seam from all the auxiliary structure which depend on it to allow efficient update
    at the next iteration. Also it moves the black/white border one row deeper in the structures. This way at the next iterations
    we still do not have to treat borders differently from interior points. """
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    rows,cols= size(img) .- (iterations, 0) 
    @inbounds for j in 1:cols
        if seam[j] != rows
            @inbounds for i in seam[j]: rows-1
                ##all of these need the seam to be removed to enable the efficiently update at the next step
                img[i, j] = img[i+1, j]
                bright[i, j] = bright[i+1, j]  
                intensity_gradient[i, j] = intensity_gradient[i+1, j] 
                moves[i, j] = moves[i+1, j] 
                aux_energy[i, j] = aux_energy[i+1, j] 
            end     
        end
    end
    ## I have to add the black border to bright and the inf border for aux energy
    @views bright[rows, :] .= 0
    @views aux_energy[rows,:] .= Inf
    aux.iterations += 1
    return
end

function seamcarve1(aux::Img_Aux)
    """First seam carving iteration"""
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    get_energy(aux)
    get_seam(aux)
    remove_seam(aux)
    return
end

function seamcarve2(aux::Img_Aux)
    """Seam carving iterations suubsequent to the first one"""
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    upd_energy(aux)
    upd_seam(aux)
    remove_seam(aux)
    return
end

function seam_carve(aux::Img_Aux, n::Int64)
    """Seam carving procedure as a whole, at the end we cut once for all the image and transpose it again."""
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    seamcarve1(aux)
    for i in 2:n
        seamcarve2(aux)
    end
    ##Now we can cut and re-transpose to give the final result.
    return aux.img[1:end-n,:]'
end




######################################################################
################# ADAPTIVE SEAM CARVE IMPLEMENTATION #################
######################################################################

function ada_seam_carve1(aux::Img_Aux, n::Int64 ; ada_size::Int64=20)
    """Adaptive seam carving chunk of iterations. First we assess if it is better to remove horizontal or vertical seams, then
    we proceed to perform "ada_size" (default 20) classical seam carving iterations. If one wishes to change the cardinality of the chunks
    of adaptive iterations, he can just set the keyword argument "ada_size" to the desired value"""
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    res,aux2 = get_direction(aux,ada_size)
    res == "rows" ? new_img=seam_carve(aux, min(n,ada_size)) : new_img=seam_carve(aux2, min(n,ada_size)) 
    return new_img
end

function ada_seam_carve(aux::Img_Aux, n::Int64; ada_size::Int64=20)
    """ Adaptive seam carve procedure as a whole. It divides the procedure into chunks of size "ada_size" (default 20). At the end of
    every chunk it cuts the image and then re-initializes the auxiliary structure."""
    ada_steps=Int(floor(n/ada_size))
    for k in 1:ada_steps
        new=ada_seam_carve1(aux, n, ada_size=ada_size)
        aux=initialize_aux(new)
    end
    ##remainder of iterations
    δ = n%ada_size
    δ > 0 && (new=ada_seam_carve1(aux, n, ada_size=δ))
    return new
end

function get_direction(aux::Img_Aux, ada_size::Int64)
    """This function assess if, given the current state, it is better to reduce the image by removing horizontal of vertical seams.
    To do so it calculates the sum of the "ada_size" minimum normalized verical and horizontal seams, and chooses the smaller one."""
    @extract aux : img bright sobel_x sobel_y intensity_gradient aux_energy moves seam iterations
    ## Also here, if possible we use efficient updating.
    if iterations == 0
        get_energy(aux)
        get_seam(aux)
    else
        upd_energy(aux)
        update_seam(aux)
    end

    ## I Do not remove
    rows,cols=size(img)
    seams_costs=aux_energy[:, cols]
    ##Now I have to account for the fact that seams might be of different lenght, so we normalize
    ## Also, we do not consider just the minimum seam, but the bottom ada_size ones as we perform this step. 
    minimum_costs = 1/cols * (sum(sort(seams_costs)[1:ada_size]))

    ##Now we have to see if we'd take out the vertical seam. The intensity gradient calculation remain the same, i can just transpose.
    aux2 = initialize_aux(img')
    @views aux2.intensity_gradient .= intensity_gradient'
    get_seam(aux2)
    ## I transpose so that we get more efficient implementation.
    rows2,cols2=size(img')
    seams_costs2=aux2.aux_energy[:,cols2]
    minimum_costs2 = 1/cols2 * (sum(sort(seams_costs2)[1:ada_size]))
    minimum_costs < minimum_costs2 ? result="rows" : result="cols"
    return (result, aux2)
end



end ##end module