"""
    rbfcentre(model::SVDD;
              minerr::T=T(1e-3),
              maxiters::Int=100,
              verbose::Bool=false) -> c
Return the estimated `c`enter of model via fixed point method.

    c = ∑ᵢ wᵢ * xᵢ, where 
    wᵢ = αᵢ / ∑ⱼαⱼ k(c, xⱼ) and xᵢ, xⱼ ∈  SupportVectors

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
    xsvs = svs(model)
    α = alphas(model)
    k = model.kernel
    N = nsvs(model)
    N⁻¹ = T(inv(N))  # normalize err, so size independent
    c = xsvs[:,1:1]  # center's start point
    err = typemax(T)
    cnt = 0
    while err > minerr
        cnt += 1
        cnt > maxiters && break
        w = α .* k(c, xsvs)
        s = (w ./ sum(w)) .* xsvs
        μ = sum(s, dims=2)
        err = sum(abs.(c - μ)) * N⁻¹
        c .= μ
        verbose && println("iter $cnt, err=$err")
    end
    return c
end


"""
    centre(model::SVDD, x::Matrix{<:AbstractFloat})
Return the nearest feature of `x` from sphere center of `model`. It's 
a tough version of centre method if you don't mind much computation.
This method is usually for the case that `x` is the training set, i.e. 
find the pre-image center in training set `x` that minimize |ϕ(xᵢ) - cᵩ|
"""
@inline centre(model::SVDD, x::Matrix{T}) where {T <: AbstractFloat} = min(model, x)




