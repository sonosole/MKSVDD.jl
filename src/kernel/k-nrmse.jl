"""
    d = √(mean(|x - y|²)) / (maximum(x) - minimum(x))
    k(x, y) = exp(- d / σ)
"""
struct NRMSEK <: XKernel
    sigma :: Real
    function NRMSEK(σ::Real)
        @assert σ > 0 "The standard deviation should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::NRMSEK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(NormRMSDeviation(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ)
    return exp.(- γ .* d)
end


