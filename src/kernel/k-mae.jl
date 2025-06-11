"""
    d = mean(|x - y|)
    k(x, y) = exp(- d / σ)
"""
struct MAEK <: XKernel
    sigma :: Real
    function MAEK(σ::Real)
        @assert σ > 0 "The input should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::MAEK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(MeanAbsDeviation(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ)
    return exp.(- γ .* d)
end


