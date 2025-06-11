"""
    d = mean(|x - y|²)
    k(x, y) = exp(- √d / σ)
"""
struct RMSEK <: XKernel
    sigma :: Real
    function RMSEK(σ::Real)
        @assert σ > 0 "The standard deviation should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::RMSEK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(RMSDeviation(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ)
    return exp.(- γ .* d)
end


