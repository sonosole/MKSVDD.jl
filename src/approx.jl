"""
    basecoeffs(k::Function,   # kernel function
               α::Matrix{T},  # N*1
               x::Matrix{T},  # d*N
               z::Matrix{T}   # d*K
               ) where T -> β::Matrix{T}  # K*1

to estimate new bases' coefficients by optimazing:

    min(J) = min‖∑ⱼβⱼ*ϕ(zⱼ) - ∑ᵢαᵢ*ϕ(xᵢ)‖²
           = min(βᵀ*Kzz*β - 2αᵀ*Kxz*β + αᵀ*Kxx*α)
           ≡ min(βᵀ*Kzz*β - 2αᵀ*Kxz*β), where β ∈ Rᴹ¹, α ∈ Rᴺ¹, 

    by ∂J/∂β = 2Kzz*β - 2Kxzᵀ*α
             = 2Kzz*β - 2Kzx*α = 0 ⇒ Kzz*β = Kzx*α, so we get:

    β = Kzz⁻¹*Kxzᵀ*α = Kzz⁻¹*Kzx*α, or 
    β = Kzz ╲ (Kzx*α), (left division operator, faster)
    where Kzz ∈ Rᴹᴹ, Kxz ∈ Rᴺᴹ

usually the more `K` the better estimation.
"""
function basecoeffs(k::Function,  # kernel function
                    α::Matrix{T}, # N*1
                    x::Matrix{T}, # d*N
                    z::Matrix{T}  # d*K
                    ) where T
    Kzz = k(z,z) # K*K
    Kzx = k(z,x) # K*N
    β = Kzz \ (Kzx * α) # K*1
    return β
end



# 从提供的特征 x 中选 K 个基向量来逼近球心，只更新 β 不更新 z
"""
    rbfksvdd(model::SVDD{T},
          features::Matrix{T},
                 K::Int;
            niters::Int=5,
           verbose::Bool=false) where {T} -> model'::SVDD{T}

Chose `K` coloumn vectors `z` from `features`' K-medoids, to estimate 
`model`'s sphere center `cᵩ` via optimazing:

    min(J) = min‖∑ᵢβᵢ*ϕ(zᵢ) - cᵩ‖², β ∈ Rᴷ¹, i=1:K

usually the more `K` the better estimation.
"""
function rbfksvdd(model::SVDD{T,N},
               features::Matrix{T},
                      K::Int;
                 niters::Int=5,
                verbose::Bool=false) where {T,N}
    # ──── select K init bases from K medoids of features ────
    𝕜 = kernelf(model)
    Δ = distmat(𝕜, features)
    c₁, L₁ = kcentreids(Δ, K; niters, verbose)
    c₂, L₂ = kcentreids(Δ, K; niters, verbose)
    cids = L₁ < L₂ ? c₁ : c₂  # chose low cost
    z = features[:, cids]

    # ──── estimate coefficients of new base ────
    α = alphas(model)   # N*1, N lagrangers
    x = svs(model)      # d*N, N support vectors
    Kzz = 𝕜(z,z)        # K*K, distmat between new bases'
    Kzx = 𝕜(z,x)        # K*N, distmat from old bases x to new bases z
    β = Kzz \ (Kzx * α) # K*1, K coefficients of new base
    
    # ──── re-estimate R² and <cᵩ, cᵩ> ────
    b   = onesv(model)
    Kbb = 𝕜(b, b) # 1*1
    Kzb = 𝕜(z, b) # K*1
    βKβ = β' * Kzz * β
    R²  = Kbb + βKβ - 2β' * Kzb # ‖ϕ(b) - ∑ᵢβᵢ*ϕ(zᵢ)‖²
    kmodel = SVDD{T,N}(first(R²), first(βKβ), reshape(β,1,:), z, 𝕜)
    if verbose
        d = dcenters(model, kmodel)
        println("── centers distance: $d ──")
    end
    return kmodel
end




# 从所有支持向量 s 中选 K 个基向量来逼近球心
"""
    rbfksvdd(model::SVDD{T,N},
                 K::Int;
            niters::Int=5,
           verbose::Bool=false) where {T,N} -> model'::SVDD{T,N}

Chose `K` coloumn vectors `z` from `model`'s support vectors, to estimate 
`model`'s sphere center `cᵩ` via optimazing:

    min(J) = min‖∑ᵢβᵢ*ϕ(zᵢ) - cᵩ‖², β ∈ Rᴷ¹, i=1:K

usually the more `K` the better estimation.
"""
function rbfksvdd(model::SVDD{T,N},
                      K::Int;
                 niters::Int=5,
                verbose::Bool=false) where {T,N}
    return rbfksvdd(model, svs(model), K; niters, verbose)
end


# 只用一个预像近似中心 min‖ϕ(z) - cᵩ‖²
function extremeprune(model::SVDD{T,N};
                   maxiters::Int=100,
                     minerr::T=T(1e-3),
                    verbose::Bool=false) where {T,N}
    𝕜 = kernelf(model)
    α = alphasᵀ(model)
    x = svs(model)
    z = rbfpreimage(𝕜, α, x; maxiters, minerr, verbose)
    b = onesv(model) # support vector on surface
    Kbb = 𝕜(b, b)    # 1*1
    Kzb = 𝕜(z, b)    # 1*1
    Kzz = 𝕜(z, z)    # 1*1
    R²  = Kbb + Kzz - 2Kzb
    return SVDD{T,N}(first(R²), first(Kzz), ones(T,1,1), z, 𝕜)
end
