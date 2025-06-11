"""
    d = 1 - ∑(min(x, y)) / ∑(max(x, y))
    k(x, y) = exp(- d / σ)
"""
struct JaccardK <: XKernel
    sigma :: Real
    function JaccardK(σ::Real)
        @assert σ > 0 "The input should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::JaccardK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(Jaccard(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ)
    return exp.(- γ .* d)
end


