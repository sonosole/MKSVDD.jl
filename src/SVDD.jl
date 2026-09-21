mutable struct SVDD{T <: AbstractFloat, N}
    R²     :: T
    wᵀKw   :: T
    𝟐wᵀ    :: Matrix{T}
    svecs  :: Matrix{T}
    kernel :: Function
    function SVDD{T,N}(R²::T,
                     wᵀKw::T,
                       wᵀ::Matrix{T},
                    svecs::Matrix{T},
                 xykernel::Function) where {T <: AbstractFloat, N}
        @assert N==1 || N==2 begin
            "only support type SVDD{T,1} or SVDD{T,2}, got SVDD{T,$N}"
        end
        new{T,N}(R², wᵀKw, lmul!(T(2),wᵀ), svecs, xykernel)
    end
end


"""
    Return the kernel function of `model`
"""
@inline kernelf(model::SVDD) = model.kernel

"""
    Return the number of support vectors
"""
@inline nsvs(model::SVDD) = length(model.𝟐wᵀ)

"""
    Return the support vectors
"""
@inline svs(model::SVDD) = model.svecs

"""
return the number of support vectors
"""
Base.length(model::SVDD) = length(model.𝟐wᵀ)


"""
    alphas(m::SVDD{T}) -> α::Matrix{T}
Return the lagrange multipliers
"""
@inline function alphas(model::SVDD{T}) where T
    return model.𝟐wᵀ .* T(0.5)
end


"""
    radius(model::SVDD{T}) -> r::T
Returns the radius of the hypersphere
"""
@inline function radius(model::SVDD)
    return sqrt(model.R²)
end

@inline function radius²(model::SVDD)
    return model.R²
end

function Base.show(io::IO, ::MIME"text/plain", model::SVDD{T,N}) where {T, N}
    C = nsvs(model)
    R = radius(model)
    print(io, "SVDD{$T,$N} with $C support vectors, radius=$R")
end


# inplace sqrt
@inline function sqrt!(x::AbstractArray)
    x .= sqrt.(x)
    return x
end


function Base.abs(model::SVDD, feat::Matrix{T}) where {T <: AbstractFloat}
    wᵀKw = model.wᵀKw
    xs   = model.svecs
    𝟐wᵀ  = model.𝟐wᵀ
    κ    = model.kernel
    N  = size(feat, 2)
    Δ² = Vector{T}(undef, N) # Δ² = ║x - c║²
    for i ∈ 1:N
        x = feat[:, i:i]
        Kxx = κ(x,  x)
        Ksx = κ(xs, x)
        Δ²[i] = first(Kxx - 𝟐wᵀ * Ksx) + wᵀKw
    end
    return sqrt!(Δ²)
end


function Base.abs2(model::SVDD, feat::Matrix{T}) where {T <: AbstractFloat}
    wᵀKw = model.wᵀKw
    xs   = model.svecs
    𝟐wᵀ  = model.𝟐wᵀ
    κ    = model.kernel
    N  = size(feat, 2)
    Δ² = Vector{T}(undef, N) # Δ² = ║x - c║²
    for i ∈ 1:N
        x = feat[:, i:i]
        Kxx = κ(x,  x)
        Ksx = κ(xs, x)
        Δ²[i] = first(Kxx - 𝟐wᵀ * Ksx) + wᵀKw
    end
    return Δ²
end


"""
    min(model::SVDD, x::Matrix{<:AbstractFloat})
Return the nearest feature of `x` from sphere center of `model`
"""
function Base.min(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    Δ² = abs2(model, x)
    i  = argmin(Δ²)
    return x[:,i:i]
end


"""
    minratio(model::SVDD, x::Matrix{<:AbstractFloat}) -> x[:,i:i], Δ[i]/R
+ `x[:,i:i]` the nearest feature (with index `i`) of `x` from sphere center of `model`
+ `Δ` is the distance of `x` away from the center of hypersphere.
+ `R` is the radius of hypersphere.
"""
function minratio(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    R² = radius²(model)
    Δ² = abs2(model, x)
    i  = argmin(Δ²)
    return x[:,i:i], sqrt(Δ²[i]/R²)
end

"""
    max(model::SVDD, x::Matrix{<:AbstractFloat})
Return the farest feature of `x` from sphere center of `model`
"""
function Base.max(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    Δ² = abs2(model, x)
    i  = argmax(Δ²)
    return x[:,i:i]
end


"""
    maxratio(model::SVDD, x::Matrix{<:AbstractFloat}) -> x[:,i:i], Δ[i]/R
+ `x[:,i:i]` the farest feature (with index `i`) of `x` from sphere center of `model`
+ `Δ` is the distance of `x` away from the center of hypersphere.
+ `R` is the radius of hypersphere.
"""
function maxratio(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    R² = radius²(model)
    Δ² = abs2(model, x)
    i  = argmax(Δ²)
    return x[:,i:i], sqrt(Δ²[i]/R²)
end


"""
    argmin(model::SVDD, x::Matrix{<:AbstractFloat}) -> index::Int
Return the nearest feature `index` of `x` from sphere center of `model`
"""
function Base.argmin(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    Δ² = abs2(model, x)
    return argmin(Δ²)
end

"""
    argmax(model::SVDD, x::Matrix{<:AbstractFloat}) -> index::Int
Return the farest feature `index` of `x` from sphere center of `model`
"""
function Base.argmax(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    Δ² = abs2(model, x)
    return argmax(Δ²)
end


"""
    absratio(model::SVDD, x::Matrix) -> Δ / R
Return `Δ / R` ∈ [0,+∞], where 
+ `Δ` is the distance of `x` away from the center of hypersphere.
+ `R` is the radius of hypersphere.
"""
function absratio(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    R = radius(model)
    Δ = abs(model, x)
    return Δ .* inv(R)
end


"""
    abs2ratio(model::SVDD, x::Matrix) -> Δ² / R²
Return `Δ² / R²` ∈ [0,+∞], where 
+ `Δ` is the distance of `x` away from the center of hypersphere.
+ `R` is the radius of hypersphere.
"""
function abs2ratio(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    R² = radius²(model)
    Δ² = abs2(model, x)
    return Δ² .* inv(R²)
end


"""
    diff(model::SVDD, x::Matrix{T}) -> (Δ .- R)
Return the result of (Δ .- R) where `Δ` is difference of `x` from the sphere center `R`.
"""
function Base.diff(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    R = radius(model)
    Δ = abs(model, x)
    return Δ .- R
end


"""
    sqdiff(model::SVDD, x::Matrix{T}) -> (Δ² .- R)
Return the result of (Δ² .- R) where `Δ` is difference of `x` from the sphere center `R`.
"""
function sqdiff(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat}
    R² = radius²(model)
    Δ² = abs2(model, x)
    return Δ² .- R²
end


"""
    svddprob(θ::SVDD, x::Matrix{<:AbstractFloat}, γ::Real=1.0f0; type::String="gaussian")
A kind of proxy probability of `P(x|θ) ∈ [0,1]`, where
+ `γ` > 0 tunes the flatness of the distribution, the smaller the flatter.
+ `Δ` is the distance away from the center of hypersphere.
+ `R` is the radius of hypersphere.
# Probability `type`
+ "gaussian",   `P(x|θ) = exp(-γ Δ²/R²)`
+ "laplace",    `P(x|θ) = exp(-γ Δ/R)`
+ "triangle",   `P(x|θ) = max(0, 1 - γ Δ/R)`
+ "sqtriangle", `P(x|θ) = max(0, 1 - γ Δ²/R²)`
+ "dirac",      `P(x|θ) = 𝟙[γΔ ≤ R]`
"""
function svddprob(model::SVDD, x::Matrix{T}, g::Real=1.0f0; type::String="gaussian") where {T <: AbstractFloat}
    o  = zero(T)
    l  = one(T)
    γ  = abs(T(g))
    R  = radius(model)
    R² = radius²(model)
    if isequal(type, "gaussian") # exp(-γ Δ²/R²)
        r = - γ / R²
        Δ² = abs2(model, x)
        return @. exp(r * Δ²)
    end
    if isequal(type, "laplace") # exp(-γ Δ/R)
        r = - γ / R
        Δ = abs(model, x)
        return @. exp(r * Δ)
    end
    if isequal(type, "triangle") # max(0, 1 - γ Δ/R)
        Δ = abs(model, x)
        r = -clamp(γ, o, l) / R
        return @. max(o, l + r * Δ)
    end
    if isequal(type, "sqtriangle") # max(0, 1 - γ Δ²/R²)
        Δ² = abs2(model, x)
        r = -clamp(γ, o, l) / R²
        return @. max(o, l + r * Δ²)
    end
    if isequal(type, "dirac") # 1 if γΔ ≤ R, otherwise 0
        r = clamp(γ, o, l)
        Δ² = abs2(model, x)
        return @. r*Δ² ≤ R²
    end
    error("$type is not supported yet")
end

