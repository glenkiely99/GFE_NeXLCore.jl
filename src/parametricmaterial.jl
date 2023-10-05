using StaticArrays
using DataStructures
using Printf

data_dir = joinpath(@__DIR__, "..", "data")


function linspace(lb, rb, n::Integer)
    h = (rb .- lb) ./ (n-1)
    return [lb .+ i*h for i in 0:(n-1)]
end

"""
    linspace(lb, rb, n::Integer)
    
Logarithmically spaced `n` points between `lb` and `rb`
"""
logspace(lb, rb, n::Integer) = exp.(linspace(log.(lb), log.(rb), n))

export_ecs_energies = logspace(50.0, 1e5, 200)

struct VoseAlias
    alias::Vector{Int}
    prob::Vector{Float64}
end

function loaddata(filenamealias::String, filenameprob::String)
    data = Vector{VoseAlias}()
    aliasdata = zeros(Int, 606, length(export_ecs_energies))
    probdata = zeros(606, length(export_ecs_energies))
    open(filenamealias, "r") do io
        read!(io, aliasdata)            
    end
    open(filenameprob, "r") do io
        read!(io, probdata)      
    end
    for i in (1:length(export_ecs_energies))
        push!(data, VoseAlias(aliasdata[:,i], probdata[:,i]))
    end 
    return data
end

materialDD = DefaultDict{Element, Vector{VoseAlias}}(passkey=true) do elm
    filenamealias = joinpath(data_dir, "penatfiles", @sprintf("penat%03d.alias", elm.number))
    filenameprob = joinpath(data_dir, "penatfiles", @sprintf("penat%03d.prob", elm.number))
    println("Reading and caching penat alias file: $filenamealias")
    loaddata(filenamealias, filenameprob)
end


function volume_conserving_density(elms::SVector{N, Element}) where N
    ρ_pure = SVector{N, Float64}(density.(elms))
    function _density(c, elms)
        return 1. / sum(c[i] / ρ_pure[i] for (i, elm) in enumerate(elms))
    end
    return _density
end

"""
Holds basic data about a material including name, composition in mass fraction and optional propreties.

By default, Material assumes nominal terrestrial atomic weights.  However, it is possible to assign custom
atomic weights on a per element-basis for non-terrestrial materials.

The mass fraction and atomic weight are immutable but the `Properties` can be modified.

    Material(
        name::AbstractString,
        massfrac::AbstractDict{Element,U},
        atomicweights::AbstractDict{Element,V} = Dict{Element,Float64}(),
        properties::AbstractDict{Symbol,Any} = Dict{Symbol,Any}(),
    ) where { U <: AbstractFloat, V <: AbstractFloat }


**Properties**

    :Density # Density in g/cm³
    :Description # Human friendly
    :Pedigree #  Quality indicator for compositional data ("SRM-XXX", "CRM-XXX", "NIST K-Glass", "Stoichiometry", "Wet-chemistry by ???", "WDS by ???", "???")
    :Conductivity = :Insulator | :Semiconductor | :Conductor
    :OtherUserProperties # Other properties can be defined as needed
"""
struct ParametricMaterial{T<:AbstractFloat, N}
    name::String
    elms::SVector{N, Element}
    # chane to svector of type T
    #massfrac::Dict{Element, T}
    massfrac::MVector{N, T}
    massfracfunc!::Function
    a::SVector{N, T} # Optional: custom atomic weights for the keys in this Material
    ρ::Function
    properties::Dict{Symbol,Any} # :PureDensities, :Description, :Pedigree, :Conductivity, ... + user defined

    """
        Material(
            name::AbstractString,
            massfrac::AbstractDict{Element,U},
            atomicweights::AbstractDict{Element,V} = Dict{Element,Float64}(),
            properties::AbstractDict{Symbol,Any} = Dict{Symbol,Any}(),
        ) where { U <: AbstractFloat, V <: AbstractFloat }
    """
    function ParametricMaterial{T}(
        name::AbstractString,
        elms::AbstractVector{Element},
        massfracfunc!::Function,
        atomicweights::Union{AbstractArray{T}, Nothing}=nothing,
        ρ::Union{Nothing, Function}=nothing,
        properties::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),
    ) where {T<:AbstractFloat}
        N = length(elms)
        elms = SVector{N, Element}(elms) # needed?
        massfrac = isnothing(massfracfunc!) ? MVector{N,T}(ones(T, N)) : MVector{N,T}(rand(T, N))
        if isnothing(ρ)
            ρ = volume_conserving_density(elms)
        end
        if isnothing(atomicweights)
            atomicweights = a.(elms)
        end
        # below is for finding the files for probabilities at given energies
        for elm in elms
            materialDD[elm]
        end
        atomicweights = SVector{N, T}(atomicweights)
        new{T,N}(name, elms, massfrac, massfracfunc!, atomicweights, ρ, properties)
    end
    
    function ParametricMaterial(args...; kwargs...)
        ParametricMaterial{Float64}(args..., kwargs...)
    end
end

function massfractions(mat::ParametricMaterial, x::AbstractArray)
    mat.massfracfunc!(mat.massfrac, mat.elms, x)
    return mat.massfrac # outputs dictionary
end

function density(mat::ParametricMaterial, x::AbstractArray)
    c = massfractions(mat, x)
    return mat.ρ(c, mat.elms)
end
function density(mat::ParametricMaterial)
    return mat.ρ(mat.massfrac, mat.elms)
end

function atoms_per_cm³(mat::ParametricMaterial) 
    return sum(atoms_per_g(elm) * mat.massfrac[i] * density(mat) for (i, elm) in enumerate(mat.elms))
end

"""
ParametricMaterial(
    "FeNi",
    [n"Fe", n"Ni"],
    massfracfunc!,
)

function matfracfunc!(massfrac::Vector, x::AbstractArray)
    massfrac[0] = sin(x[1])
    massfrac[1] = 1 - massfrac[0]
end
"""


