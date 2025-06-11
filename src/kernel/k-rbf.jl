"""
    k(x, y) = exp(- ∑ᵢ|xᵢ - yᵢ|² / 2σ²)
"""
struct RBFK <: XKernel
    sigma :: Real
    function RBFK(σ::Real)
        @assert σ > 0 "The standard deviation should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::RBFK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(SqEuclidean(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(2σ*σ)
    return exp.(- γ .* d)
end


