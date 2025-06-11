"""
    k(x, y) = exp(- max|x - y| / σ)
"""
struct ChebyK <: XKernel
    sigma :: Real
    function ChebyK(σ::Real)
        @assert σ > 0 "The input should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::ChebyK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(Chebyshev(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ)
    return exp.(- γ .* d)
end


