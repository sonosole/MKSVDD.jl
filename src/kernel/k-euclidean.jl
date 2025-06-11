"""
    k(x, y) = exp(- √(∑ᵢ|xᵢ - yᵢ|²) / √2σ)
"""
struct EucK <: XKernel
    sigma :: Real
    function EucK(σ::Real)
        @assert σ > 0 "The standard deviation should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::EucK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(Euclidean(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(√2σ*σ)
    return exp.(- γ .* d)
end


