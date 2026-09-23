"""
    rbfpreimage(k::Function,  # RBF kernel function
                α::Matrix{T}, # coefficients, shape of 1*N
                x::Matrix{T}; # feature, with shape of D*N
                minerr::T=T(1e-3),
                maxiters::Int=100,
                verbose::Bool=false) where T -> z
Return the estimated pre-imagge via fixed point method by solving

    min‖ϕ(z) - ∑ⱼαⱼ ϕ(xⱼ)‖², so
    z = ∑ᵢ wᵢ * xᵢ, where 
    wᵢ = αᵢ / ∑ⱼαⱼ k(z, xⱼ) and xᵢ, xⱼ ∈  x
    if ∑ⱼαⱼ = 1, then ϕ(z) is the center of inputs ϕ.(x), otherwise it's not center

`z` is iterated until it doesn't change much by `minerr` or `maxiters`. If 
`verbose` then print the iteration process.
!!! note
    This is only for RBF kernel `exp(-‖x-y‖²/σ²)`. When the kernel width `σ` is 
    + very large, kernel landscape is very flat, unstable but not changed much
    + very small, kernel landscape is steep, stable but changed quickly
    so `z` might be unstable and not unique.
"""
function rbfpreimage(k::Function,
                     α::Matrix{T},
                     x::Matrix{T};
                     minerr::T=T(1e-3),
                     maxiters::Int=100,
                     verbose::Bool=false) where T
    N = size(x, 2)
    rα, cα = size(α)
    @assert rα == 1 "the coefficients shall be a 1*N shaped matrix"
    @assert cα == N "#coefficients doesn't match #features"
    N⁻¹ = T(inv(N))  # normalize err, so size independent
    j = rand(1:N)
    z = x[:,j:j]     # a random start from given samples
    err = typemax(T)
    cnt = 0
    verbose && println("─────── iter pre-image process ────────")
    while err > minerr
        cnt += 1
        cnt > maxiters && break
        w = α .* k(z, x)
        s = (w ./ sum(w)) .* x
        μ = sum(s, dims=2)
        err = sum(abs.(z - μ)) * N⁻¹
        z .= μ
        verbose && println("iter $cnt, err=$err")
    end
    return z
end




