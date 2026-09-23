# 从提供的特征 x 中选 K 个基向量来逼近球心
"""
    rbfksvdd(model::SVDD{T,N},
                 x::Matrix{T},
                 K::Int;
            niters::Int=5,
           verbose::Bool=false) where {T,N} -> model'::SVDD{T,N}

Chose `K` coloumn vectors `z` from `x`'s K-medoids, to estimate 
`model`'s sphere center via optimazing:

    min(J) = min‖∑ᵢβᵢ*ϕ(zᵢ) - ∑ⱼαⱼ*ϕ(sⱼ)‖², z ∈ x
        = min(βᵀ*Kzz*β - 2αᵀ*Ksz*β + αᵀ*Kss*α)
        ≡ min(βᵀ*Kzz*β - 2αᵀ*Ksz*β)

    by ∂J/∂β = 2Kzz*β - 2Kszᵀ*α
             = 2Kzz*β - 2Kzs*α = 0 ⇒ Kzz*β = Kzs*α

    β = Kzz⁻¹*Kszᵀ*α = Kzz⁻¹*Kzs*α, or 
    β = Kzz ╲ (Kzs*α), (left division operator, faster)
    β ∈ Rᴷ¹, α ∈ Rᴺ¹, Kzz ∈ Rᴷᴷ, Ksz ∈ Rᴺᴷ

the more `K` the better estimation.
"""
function rbfksvdd(model::SVDD{T,N},
                      x::Matrix{T},
                      K::Int;
                 niters::Int=5,
                verbose::Bool=false) where {T,N}
    𝕜 = kernelf(model)
    s = svs(model)
    Δ = distmat(𝕜, x)
    c₁, L₁ = kcentreids(Δ, K; niters, verbose)
    c₂, L₂ = kcentreids(Δ, K; niters, verbose)
    z = x[:, L₁ < L₂ ? c₁ : c₂] # chose low cost
    α = alphas(model)   # N*1
    Kzz = 𝕜(z, z)       # K*K
    Kzs = 𝕜(z, s)       # K*N
    β = Kzz \ (Kzs * α) # K*1
    b = onesv(model)
    Kbb = 𝕜(b, b) # 1*1
    Kzb = 𝕜(z, b) # K*1
    βKβ = β' * Kzz * β
    R²  = Kbb + βKβ - 2β' * Kzb
    return SVDD{T,N}(first(R²), first(βKβ), reshape(β,1,:), z, 𝕜)
end




# 从所有支持向量 s 中选 K 个基向量来逼近球心
"""
    rbfksvdd(model::SVDD{T,N},
                 K::Int;
            niters::Int=5,
           verbose::Bool=false) where {T,N} -> model'::SVDD{T,N}

Chose `K` coloumn vectors `z` from `model`'s support vectors, to estimate 
`model`'s sphere center via optimazing:

    min(J) = min‖∑ᵢβᵢ ϕ(zᵢ) - ∑ⱼαⱼ ϕ(sⱼ)‖², z ∈ s
        = min(βᵀ*Kzz*β - 2αᵀ*Ksz*β + αᵀ*Kss*α)
        ≡ min(βᵀ*Kzz*β - 2αᵀ*Ksz*β)

    by ∂J/∂β = 2Kzz*β - 2Kszᵀ*α
             = 2Kzz*β - 2Kzs*α = 0 ⇒ Kzz*β = Kzs*α

    β = Kzz⁻¹*Kszᵀ*α = Kzz⁻¹*Kzs*α, or 
    β = Kzz ╲ (Kzs*α), (left division operator, faster)
    β ∈ Rᴷ¹, α ∈ Rᴺ¹, Kzz ∈ Rᴷᴷ, Ksz ∈ Rᴺᴷ

the more `K` the better estimation.
"""
function rbfksvdd(model::SVDD{T,N},
                      K::Int;
                 niters::Int=5,
                verbose::Bool=false) where {T,N}
    return rbfksvdd(model, svs(model), K; niters, verbose)
end
