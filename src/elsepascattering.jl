using StaticArrays
"""
Alias table method for finding the appropriate scattering angle. 
"""
abstract type ELSEPAScatteringCrossSectionType <: ElasticScatteringCrossSection end

struct ELSEPAScatteringCrossSection <: ELSEPAScatteringCrossSectionType end

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
        return energies[1]
    end
    if rand() < (log(E) - log(energies[i-1])) / (log(energies[i]) - log(energies[i-1])) #loglog interpolate
        return energies[i-1]
    else
        return i, energies[i]
    end
end

function δσδΩ(::Type{ELSEPAScatteringCrossSection}, θ::Float64, elm::Element, E::Float64)::Float64
    i, - = interpolateE(ELSEPAScatteringCrossSection, E)
    probgrid = parametricDD[elm][i] # query static vector index to return static vector of probabilities 
end


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

    return res
end