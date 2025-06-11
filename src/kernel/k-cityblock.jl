"""
    k(x, y) = exp(- ∑ᵢ|xᵢ - yᵢ| / σ)
"""
struct CityK <: XKernel
    sigma :: Real
    function CityK(σ::Real)
        @assert σ > 0 "The input should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::CityK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(Cityblock(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ)
    return exp.(- γ .* d)
end


