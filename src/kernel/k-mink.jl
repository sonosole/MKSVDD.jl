"""
    k(x, y) = exp(- ᵖ√(∑ᵢ|xᵢ - yᵢ|ᵖ) / σ)
"""
struct MinkowskiK <: XKernel
    sigma :: Real
    p     :: Real
    function MinkowskiK(σ::Real, p::Real=1)
        @assert σ > 0 "The standard deviation should be greater than zero, got $σ"
        new(σ, p)
    end
end


function kmat(kernel::MinkowskiK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    σ = T(kernel.sigma)
    p = T(kernel.p)
    γ = inv(σ)
    d = pairwise(Minkowski(p), x, y, dims=obsdim)
    return exp.(- γ .* d)
end

