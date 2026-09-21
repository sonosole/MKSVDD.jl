"""
    rbfcentre(model::SVDD;
              minerr::T=T(1e-3),
              maxiters::Int=100,
              verbose::Bool=false) -> c
Return the estimated `c`enter of model via fixed point method by solving

    min‖ϕ(c) - ∑ⱼαⱼ ϕ(xⱼ)‖², so
    c = ∑ᵢ wᵢ * xᵢ, where 
    wᵢ = αᵢ / ∑ⱼαⱼ k(c, xⱼ) and xᵢ, xⱼ ∈  SupportVectors
    s.t. ∑ⱼαⱼ = 1, otherwise it's not center

`c` is iterated until it doesn't change much by `minerr` or `maxiters`. If 
`verbose` then print the iteration process.
!!! note
    This is only for RBF kernel `exp(-‖x-y‖²/σ²)`. When the kernel width `σ` is 
    + very large, kernel landscape is very flat, unstable but not changed much
    + very small, kernel landscape is steep, stable but changed quickly
    so `c` might be unstable and not unique.
"""
function rbfcentre(model::SVDD{T};
                   minerr::T=T(1e-3),
                   maxiters::Int=100,
                   verbose::Bool=false) where T
    x = svs(model)
    α = alphas(model)
    k = kernelf(model)
    return rbfpreimage(k, α, x; minerr, maxiters, verbose)
end


"""
    centre(model::SVDD, x::Matrix{<:AbstractFloat})
Return the nearest feature of `x` from sphere center of `model`. It's 
a tough version of centre method if you don't mind much computation.
This method is usually for the case that `x` is the training set, i.e. 
find the pre-image center in training set `x` that minimize |ϕ(xᵢ) - cᵩ|
"""
@inline centre(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat} = min(model, x)




