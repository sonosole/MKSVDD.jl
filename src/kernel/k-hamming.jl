"""
    k(x, y) = exp(- ∑ᵢ(xᵢ ≠ yᵢ) / σ)
"""
struct HammingK <: XKernel
    sigma :: Real
    function HammingK(σ::Real)
        @assert σ > 0 "input should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::HammingK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(Hamming(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ)
    return exp.(- γ .* d)
end


