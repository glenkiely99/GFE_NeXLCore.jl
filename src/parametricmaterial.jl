using StaticArrays
using DataStructures

struct CSData
    values::SVector{606, Float64} # 606 values for angles and probabilities
end

# Creates a dictionary of static vectors containing the probabilities of angles
function loaddata(filename::String)
    data = Dict{Float64, CSData}()
    open(filename, "r") do f
        energy = 0.0
        values = Float64[]
        while !eof(f) # read through the file
            line = readline(f)
            if startswith(line, "-1")  
                if !isempty(values)
                    data[energy] = CSData(values)
                    values = Float64[]
                end
                parts = split(line)
                energy = parse(Float64, parts[3])
            else
                append!(values, parse.(Float64, split(line)))
            end
        end
        if !isempty(values)
            data[energy] = CSData(values)
        end
    end
    return data
end

parametricDD = DefaultDict{Element, CSData}(passkey=true) do elm
    filename = @sprintf("../data/elsepafiles/eeldx%03d.p08", elm.number)
    println("Reading and caching file: $filename")
    loaddata(filename)
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
            parametricDD[elm]
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


