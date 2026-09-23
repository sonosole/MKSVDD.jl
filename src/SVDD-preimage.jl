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
    z = getcol(x,j)  # a random start from given samples
    err = typemax(T)
    cnt = 0
    verbose && println("─────── iter pre-image process ────────")
    while err > minerr
        cnt += 1
        cnt > maxiters && break
        w = α .* k(z, x)
        ∑ = sum(w)
        s = (w .* inv(∑)) .* x
        μ = sum(s, dims=2)
        err = sum(abs.(z - μ)) * N⁻¹
        z .= μ
        verbose && println("iter $cnt, err=$err")
    end
    return z
end



"""
    trackx2y(kernel::Function, x::Matrix{T}, y::Matrix{T}, n::Int=100) -> z::Matrix{T}
Return the trace of `x` → `y` according to the trace of `ϕ(x)` → `ϕ(y)`, 
the path is stored in `z`'s `1:n` columns. `x` and `y` are both one column with 
the same dimentions. The path in `ϕ()` space is evenly spaced but usually nonlinearly 
in the original euclidean space.

# Example
```julia
begin
    kf(x::Matrix, y::Matrix) = exp.(-.1pairwise(SqEuclidean(), x, y, dims=2))
    x1 = reshape([-1.2,-1.4],:,1)
    x2 = reshape([1.5,  0.5],:,1)
    ns = 55
    xs = trackx2y(kf, x1, x2, ns)
    scatter(xs[1,:],xs[2,:], label="trace", 
                              framestyle=:origin,
                              color=:green,
                              markershape=:circle,
                              markersize=1.8,
                              markerstrokewidth=0)
end
```
"""
function trackx2y(k::Function, x::Matrix{T}, y::Matrix{T}, n::Int=100) where T
    xr,xc = size(x); @assert xc==1 "it's not a src point"
    yr,yc = size(y); @assert yc==1 "it's not a dst point"
    @assert xr==yr "dimention mismatch"
    u = hcat(x, y)
    a = range(0.0, 1.0, n)
    α = zeros(1, 2)
    z = similar(x, xr, n)
    l = one(T)
    for (i, a) ∈ enumerate(range(0.0, 1.0, n))
        α[1] = l - a
        α[2] = a
        z[:,i:i] .= rbfpreimage(k, α, u)
    end
    return z
end

