"""
    d = mean(|x - y|²)
    k(x, y) = exp(- d / σ²)
"""
struct MSEK <: XKernel
    sigma :: Real
    function MSEK(σ::Real)
        @assert σ > 0 "The standard deviation should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::MSEK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(MeanSqDeviation(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ*σ)
    return exp.(- γ .* d)
end


