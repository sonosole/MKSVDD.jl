# 矩阵对角线正则，防止逆矩阵计算数值错误
"""
    λI₊(x::AbstractMatrix, λ=1e-3)

Return `λI + x`, so `inv(λI + x)` is solvable
+ `I` is an identity matrix (main diaganal elements are ones)
+ `λ` is a small value like 1e-5
"""
function λI₊(x::AbstractMatrix{T}, λ::T=T(1e-3)) where T
    n = minimum(size(x))
    @inbounds @simd for i ∈ 1:n
        x[i,i] += λ
    end
    return x
end


# 全矩阵迭代 β/z , 存在数值不稳定的问题，所以迭代要及时退出
"""
    rbfrebase(k::Function,
              α::Matrix{T}, # N*1
              x::Matrix{T}, # d*N
              z::Matrix{T}  # d*K
              ) where T -> β, z̃

Find new base {(βⱼ, ϕ(zⱼ)) | j=1:M} to replace old base {(αᵢ, ϕ(xᵢ)) | i=1:N}, 
where `z` is `M` coloumn vectors as initials, its an optimazing problem:

    min(J) = min‖∑ⱼβⱼ*ϕ(zⱼ) - ∑ᵢαᵢ*ϕ(xᵢ)‖²
           = min(βᵀ*Kzz*β - 2αᵀ*Kxz*β + αᵀ*Kxx*α)
           ≡ min(βᵀ*Kzz*β - 2αᵀ*Kxz*β), where β ∈ Rᴹ¹, α ∈ Rᴺ¹, Kzz ∈ Rᴹᴹ, Kxz ∈ Rᴺᴹ

    (1): by ∂J/∂β = 2Kzz*β - 2Kxzᵀ*α
                 = 2Kzz*β - 2Kzx*α = 0 ⇒ Kzz*β = Kzx*α

    β = Kzz⁻¹*Kxzᵀ*α = Kzz⁻¹*Kzx*α, or 
    β = Kzz ╲ (Kzx*α), (left division operator, faster)

    (2): by ∂J/∂zₖ = 0, if and only if 𝒌 is RBF kernel

         ∑ᵢᴺ xᵢ * αᵢ𝒌(xᵢ,zₖ) - ∑ⱼᴹ zⱼ * βⱼ𝒌(zⱼ,zₖ)
    zₖ = ─────────────────────────────────────────, or its matrix version
            ∑ᵢᴺ αᵢ 𝒌(xᵢ,zₖ) - ∑ⱼᴹ βⱼ 𝒌(zⱼ,zₖ)

         x * (α .* 𝒌(x,z)) -  z * (β .* 𝒌(z,z))
    z = ─────────────────────────────────────────
              αᵀ * 𝒌(x,z)  -       βᵀ * 𝒌(z,z)

the more `K` the better estimation.
"""
function rbfrebase(k::Function,
                   α::Matrix{T}, # N*1
                   x::Matrix{T}, # d*N
                   z::Matrix{T}  # d*K
                   ) where T
    Kzz = k(z,z)
    Kxz = k(x,z)
    Kzx = transpose(Kxz)

    # ──── fix z update β once ────
    β = λI₊(Kzz, 1e-3) \ (Kzx * α) # K*1
    # ──── fix β update z once ────
    αᵀKxz = α' * Kxz
    βᵀKzz = β' * Kzz
    denominator = αᵀKxz - βᵀKzz
    if any(v->abs(v)<1e-5, denominator)
        # αᵀKxz ≈ βᵀKzz 遇到数值精度不够的问题
        return β, z
    end
    z̃ = (x * (α .* Kxz) - z * (β .* Kzz)) ./ max.(denominator, T(1e-5))
    return β, z̃
end


# 矩阵迭代 β, 而 z 逐个迭代更新, 也存在数值不稳定的问题，所以迭代要及时退出
function rbfrebaseiter(𝕜::Function,
                   α::Matrix{T}, # N*1
                   x::Matrix{T}, # d*N
                   z::Matrix{T}  # d*K
                   ) where T
    Kzz = 𝕜(z,z)
    Kzx = 𝕜(z,x)
    # ──── fix z update β once ────
    β = λI₊(Kzz, 1e-3) \ (Kzx * α) # K*1
    # ──── fix β update z once ────
    z̃ = deepcopy(z)
    for k = 1:size(z, 2)
        zₖ = getcol(z̃,k)
        Kxzₖ = 𝕜(x, zₖ)
        Kzzₖ = 𝕜(z̃, zₖ)
        αKxzₖ = α .* Kxzₖ
        βKzzₖ = β .* Kzzₖ
        denominator = sum(αKxzₖ) - sum(βKzzₖ)
        # αKxzₖ ≈ βKzzₖ 遇到数值精度不够的问题
        abs(denominator) < 1e-5 && continue
        z̃[:,k] .= (x * αKxzₖ - z * βKzzₖ) .* inv(denominator)
    end
    return β, z̃
end



# 从提供的特征 x 中选 K 个聚类中心基向量作为初始值来交替迭代 z β 以逼近球心
function rbfrebase(model::SVDD{T,N},
                features::Matrix{T},
                       K::Int;
                   iterz::Bool=true,
                  niters::Int=5,
                maxiters::Int=100,
                  minerr::T=T(1e-5),
                 verbose::Bool=false) where {T,N}
    # ──── select K init bases z from K medoids of features ────
    𝕜 = kernelf(model)
    Δ = distmat(𝕜, features)
    c₁, L₁ = kcentreids(Δ, K; niters, verbose)
    c₂, L₂ = kcentreids(Δ, K; niters, verbose)
    cids = L₁ < L₂ ? c₁ : c₂
    z = features[:, cids] # chose low cost
    β = rand(K, 1)
    
    # ──── update base z and its coefficients β ────
    verbose && println("─────── iter rebase process ────────")
    cnt = 0
    err = typemax(T)
    K⁻¹ = T(inv(K))
    n⁻¹ = T(inv(length(z)))
    α = alphas(model)   # N*1
    x = svs(model)      # d*N
    while err > minerr
        cnt += 1
        cnt > maxiters && break
        # ──── updata z β once ────
        βₑ, zₑ = iterz ? rbfrebaseiter(𝕜, α, x, z) : rbfrebase(𝕜, α, x, z)
        zerr = sum(abs.(z - zₑ)) * n⁻¹
        βerr = sum(abs.(β - βₑ)) * K⁻¹
        z = zₑ
        β = βₑ
        err = zerr
        if verbose
            # ‖∑ⱼβⱼ*ϕ(zⱼ) - ∑ᵢαᵢ*ϕ(xᵢ)‖
            δ = β'*𝕜(z,z)*β + α'*𝕜(x,x)*α - 2α'*𝕜(x,z)*β
            d = sqrt(abs(first(δ)))
            println("iter $cnt, err=$err, Δz:$zerr, Δβ:$βerr, Δcenter=$d")
        end
    end

    # ──── re-estimate R² and <cᵩ, cᵩ> ────
    b   = onesv(model)
    Kbb = 𝕜(b, b) # 1*1
    Kzb = 𝕜(z, b) # K*1
    βKβ = β' * 𝕜(z,z) * β
    R²  = Kbb + βKβ - 2β' * Kzb

    return SVDD{T,N}(first(R²), first(βKβ), reshape(β,1,:), z, 𝕜)
end

