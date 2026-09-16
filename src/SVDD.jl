mutable struct SVDD{T <: AbstractFloat, N}
    R²     :: T
    wᵀKw   :: T
    𝟐w     :: Matrix{T}
    svecs  :: Matrix{T}
    kernel :: Function
    function SVDD{T,N}(R²::T,
                     wᵀKw::T,
                        w::Matrix{T},
                    svecs::Matrix{T},
                 xykernel::Function) where {T <: AbstractFloat, N}
        @assert N==1 || N==2 begin
            "only support type SVDD{T,1} or SVDD{T,2}, got SVDD{T,$N}"
        end
        new{T,N}(R², wᵀKw, lmul!(T(2),w), svecs, xykernel)
    end
end


"""
return the number of support vectors
"""
@inline nsvs(S::SVDD) = length(S.𝟐w)


"""
return the number of support vectors
"""
Base.length(S::SVDD) = length(S.𝟐w)


"""
    alphas(m::SVDD{T}) -> α::Matrix{T}
Return the lagrange multipliers
"""
function alphas(m::SVDD{T}) where T
    return m.𝟐w .* T(0.5)
end


"""
    radius(m::SVDD{T}) -> r::T
Returns the radius of the hypersphere
"""
function radius(m::SVDD)
    return sqrt(m.R²)
end


function Base.show(io::IO, ::MIME"text/plain", svdd::SVDD{T,N}) where {T, N}
    C = nsvs(svdd)
    R = radius(svdd)
    print(io, "SVDD{$T,$N} with $C support vectors, radius=$R")
end


# inference functor
function (Model::SVDD)(feat::Matrix{T}) where {T <: AbstractFloat}
    wᵀKw = Model.wᵀKw
    xs   = Model.svecs
    R²   = Model.R²
    𝟐w   = Model.𝟐w
    κ    = Model.kernel
    N  = size(feat, 2)
    Δ² = Vector{T}(undef, N)
    for i ∈ 1:N
        x = feat[:, i:i]
        Kxx = κ(x,  x)
        Ksx = κ(xs, x)
        Δ²[i] = first(Kxx - 𝟐w' * Ksx) + wᵀKw
    end
    # Δ² .> R² ⇒ out of sphere
    # Δ² .≡ R² ⇒ on sphere surface
    # Δ² .< R² ⇒ inside sphere
    return Δ² .- R²
end


"""
    svddprob(θ::SVDD, x::Matrix{<:AbstractFloat}, γ::Real=1.0f0)
A kind of proxy probability of `p(x|θ) = exp(-γ Δ²/R²)`, where
+ `γ` > 0 tunes the flatness of the distribution, the smaller the flatter.
+ `Δ` is the distance away from the center of hypersphere.
+ `R` is the radius of hypersphere.
"""
function svddprob(Model::SVDD, feat::Matrix{T}, r::Real=1.0f0) where {T <: AbstractFloat}
    wᵀKw = Model.wᵀKw
    xs   = Model.svecs
    R²   = Model.R²
    𝟐w   = Model.𝟐w
    κ    = Model.kernel
    N  = size(feat, 2)
    Δ² = Vector{T}(undef, N)
    for i ∈ 1:N
        x = feat[:, i:i]
        Kxx = κ(x,  x)
        Ksx = κ(xs, x)
        Δ²[i] = first(Kxx - 𝟐w' * Ksx) + wᵀKw
    end
    γ = T(-abs(r)) / R²
    return exp.(γ .* Δ²)
end
