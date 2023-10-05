using StaticArrays
using NeXLCore : ElasticScatteringCrossSection
"""
Alias table method for finding the appropriate scattering angle. 
"""
abstract type ELSEPAScatteringType <: ElasticScatteringCrossSection end

struct ELSEPAScattering <: ELSEPAScatteringCrossSectionType end

const energies = [
    5.000E+01,
    6.000E+01,
    7.000E+01,
    8.000E+01,
    9.000E+01,
    1.000E+02,
    1.250E+02,
    1.500E+02,
    1.750E+02,
    2.000E+02,
    2.500E+02,
    3.000E+02,
    3.500E+02,
    4.000E+02,
    4.500E+02,
    5.000E+02,
    6.000E+02,
    7.000E+02,
    8.000E+02,
    9.000E+02,
    1.000E+03,
    1.250E+03,
    1.500E+03,
    1.750E+03,
    2.000E+03,
    2.500E+03,
    3.000E+03,
    3.500E+03,
    4.000E+03,
    4.500E+03,
    5.000E+03,
    6.000E+03,
    7.000E+03,
    8.000E+03,
    9.000E+03,
    1.000E+04,
    1.250E+04,
    1.500E+04,
    1.750E+04,
    2.000E+04,
    2.500E+04,
    3.000E+04,
    3.500E+04,
    4.000E+04,
    4.500E+04,
    5.000E+04,
    6.000E+04,
    7.000E+04,
    8.000E+04,
    9.000E+04,
    1.000E+05,
    1.250E+05,
    1.500E+05,
    1.750E+05,
    2.000E+05,
    2.500E+05,
    3.000E+05,
    3.500E+05,
    4.000E+05,
    4.500E+05,
    5.000E+05,
    6.000E+05,
    7.000E+05,
    8.000E+05,
    9.000E+05,
    1.000E+06,
    1.250E+06,
    1.500E+06,
    1.750E+06,
    2.000E+06,
    2.500E+06,
    3.000E+06,
    3.500E+06,
    4.000E+06,
    4.500E+06,
    5.000E+06,
    6.000E+06,
    7.000E+06,
    8.000E+06,
    9.000E+06,
    1.000E+07,
    1.250E+07,
    1.500E+07,
    1.750E+07,
    2.000E+07,
    2.500E+07,
    3.000E+07,
    3.500E+07,
    4.000E+07,
    4.500E+07,
    5.000E+07,
    6.000E+07,
    7.000E+07,
    8.000E+07,
    9.000E+07,
    1.000E+08,
] # eV
# energies taken from the first eeldx file! 

"""
Generates grid of 606 angles between 0 and 180 degrees
"""
function elsepa_angles()
    res = Vector{Float64}()
    push!(res, 0.0)
    push!(res, 1e-4)
    while res[end] < 180
        if res[end] < 0.9999e-3
            push!(res, res[end] + 2.5e-5)

        elseif res[end] < 0.9999e-2
            push!(res, res[end] + 2.5e-4)

        elseif res[end] < 0.9999e-1
            push!(res, res[end] + 2.5e-3)

        elseif res[end] < 0.9999e+0
            push!(res, res[end] + 2.5e-2)

        elseif res[end] < 0.9999e+1
            push!(res, res[end] + 1.0e-1)

        elseif res[end] < 2.4999e+1
            push!(res, res[end] + 2.5e-1)

        else
            push!(res, res[end] + 5.0e-1)

        end
    end
    return SVector{length(res), Float64}(res...)
end

const anglegrid = elsepa_angles()

function bSearch(E)
    low, high = 1, length(energies)
    while low <= high
        middle = div(low + high, 2)
        if energies[middle] < E
            low = middle + 1
        else
            high = middle - 1
        end
    end
    return low
end

function interpolateE(E)
    i = bSearch(E)
    if i == 1 # to account for i being the first energy
        return 1, energies[1]
    end
    if rand() < (log(E) - log(energies[i-1])) / (log(energies[i]) - log(energies[i-1])) #loglog interpolate
        return i-1, energies[i-1]
    else
        return i, energies[i]
    end
end

struct VoseAlias
    alias::Vector{Int}
    prob::Vector{Float64}
end

function VoseAlias(prob::CSData) # actually prob data
    n = length(prob.values)
    scaled_prob = prob.values .* n
    small = []
    large = []
    alias = zeros(Int, n)
    for (i, p) in enumerate(scaled_prob)
        if p < 1.0
            push!(small, i)
        else
            push!(large, i)
        end
    end
    while !isempty(small) && !isempty(large)
        l, g = pop!(small), pop!(large)
        alias[l] = g
        scaled_prob[g] -= (1.0 - scaled_prob[l])
        if scaled_prob[g] < 1.0
            push!(small, g)
        else
            push!(large, g)
        end
    end
    return VoseAlias(alias, scaled_prob) #rescales prob each time also larger than 1 due to numerical
end

function draw_sample(va::VoseAlias)
    n = length(va.prob)
    i = rand(1:n)
    if rand() < va.prob[i]
        return i
    else
        return va.alias[i]
    end
end


function NeXLCore.δσδΩ(::Type{ELSEPAScatteringCrossSection}, θ::Float64, elm::Element, E::Float64)::Float64
    i, ene = interpolateE(E)
    probgrid = parametricDD[elm][i] # query vector index to return vector of probabilities 
    va = VoseAlias(probgrid)
    angle_sample = anglegrid[draw_sample(va)]
end

function loadprobs(filenameprob::String, index::Int)
    probdata = zeros(606, length(export_ecs_energies))
    open(filenameprob, "r") do io
        read!(io, probdata)      
    end
    return probdata[:,index]
end

function sample_indices(probabilities::Vector{Float64}, n::Int)
    return sample(1:length(probabilities), Weights(probabilities), n)
end

function extract_angles(anglegrid::SVector{606, Float64}, indices::Vector{Int64})
    return anglegrid[indices]
end

function plot_histogram(angles::Vector{Float64})
    histogram(angles, xlabel="Angles", ylabel="Frequency", title="Histogram of Sampled Angles", xlims=(0,10))
end

va_angles = [anglegrid[draw_sample(parametricDD[n"Cr"][139])] for _ in 1:10_000]

function read_exported_ecs(z::Integer)
    result = zeros(606, length(export_ecs_energies))
    path = joinpath(data_dir, @sprintf("penecs%03d.bin", z))
    open(path, "r") do io
        read!(io, result)
    end
    return result
end

csdata = read_exported_ecs(24)[:,139]