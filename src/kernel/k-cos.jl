"""
    d = 1 - x ⋅ y / (‖x‖*‖y‖)
    k(x, y) = exp(- d / σ)
"""
struct CosK <: XKernel
    sigma :: Real
    function CosK(σ::Real)
        @assert σ > 0 "The input should be greater than zero, got $σ"
        new(σ)
    end
end


function kmat(kernel::CosK, x::AbstractMatrix{T}, y::AbstractMatrix{T}; obsdim::Int=2) where T <: Real
    d = pairwise(CosineDist(), x, y, dims=obsdim)
    σ = T(kernel.sigma)
    γ = inv(σ)
    return exp.(- γ .* d)
end


