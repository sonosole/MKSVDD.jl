using Mira

function sqeuc(x::Info{T}; dims::Int=2) where {T <: AbstractMatrix}
    x̃ = ᵈ(x)
    d = pairwise(SqEuclidean(), x̃; dims)
    z = Info{T}(d, x.backprop)
    if z.backprop
        z.vjp = function ∇sqeucxx()
            Δ = δ(z)
            A = Δ .+ transpose(Δ)
            𝟐 = eltype(T)(2)
            δx = if dims == 1
                𝟐 .* (sum(A; dims=2) .* x̃ .- A * x̃)
            else
                𝟐 .* (x̃ .* sum(A; dims=1) .- x̃ * A)
            end
            needgrad(x) && (x ← δx)
        end
        addkid(z, x)
    end
    return z
end


function sqeuc(x::Info{T1}, y::Info{T2}; dims::Int=2) where {T1 <: AbstractMatrix, T2 <: AbstractMatrix}
    x̃ = ᵈ(x)
    ỹ = ᵈ(y)
    T = vartype(T1, T2)
    d = pairwise(SqEuclidean(), x̃, ỹ; dims)
    z = Info{T}(d, x.backprop || y.backprop)
    if z.backprop
        z.vjp = function ∇sqeucxy()
            Δ = δ(z)
            Δᵀ = transpose(Δ)
            𝟐 = eltype(T)(2)
            needgrad(x) && begin
                addkid(z, x)
                x ← if dims == 1
                    𝟐 .* (sum(Δ; dims=2) .* x̃ .- Δ * ỹ)
                else
                    𝟐 .* (x̃ .* sum(Δᵀ; dims=1) .- ỹ * Δᵀ)
                end
            end
            needgrad(y) && begin
                addkid(z, y)
                y ← if dims == 1
                    𝟐 .* (sum(Δᵀ; dims=2) .* ỹ .- Δᵀ * x̃)
                else
                    𝟐 .* (ỹ .* sum(Δ; dims=1) .- x̃ * Δ)
                end
            end
        end
    end
    return z
end


function sqeuc(x::Info{T1}, ỹ::T2; dims::Int=2) where {T1 <: AbstractMatrix, T2 <: AbstractMatrix}
    x̃ = ᵈ(x)
    T = vartype(T1, T2)
    d = pairwise(SqEuclidean(), x̃, ỹ; dims)
    z = Info{T}(d, x.backprop)
    if z.backprop
        z.vjp = function ∇sqeucxy()
            needgrad(x) && begin
                addkid(z, x)
                Δ = δ(z)
                Δᵀ = transpose(Δ)
                𝟐 = eltype(T)(2)
                x ← if dims == 1
                    𝟐 .* (sum(Δ; dims=2) .* x̃ .- Δ * ỹ)
                else
                    𝟐 .* (x̃ .* sum(Δᵀ; dims=1) .- ỹ * Δᵀ)
                end
            end
        end
    end
    return z
end


function sqeuc(x̃::T1, y::Info{T2}; dims::Int=2) where {T1 <: AbstractMatrix, T2 <: AbstractMatrix}
    ỹ = ᵈ(y)
    T = vartype(T1, T2)
    d = pairwise(SqEuclidean(), x̃, ỹ; dims)
    z = Info{T}(d, y.backprop)
    if z.backprop
        z.vjp = function ∇sqeucxy()
            needgrad(y) && begin
                addkid(z, y)
                Δ = δ(z)
                Δᵀ = transpose(Δ)
                𝟐 = eltype(T)(2)
                y ← if dims == 1
                    𝟐 .* (sum(Δᵀ; dims=2) .* ỹ .- Δᵀ * x̃)
                else
                    𝟐 .* (ỹ .* sum(Δ; dims=1) .- x̃ * Δ)
                end
            end
        end
    end
    return z
end



# β 解析求解, z 梯度求解
function rebase(γ::T,         # k(x,y;γ) = exp(-γ‖x - y‖²)
                α::Matrix{T}, # N*1
                x::Matrix{T}, # d*N
                z::Matrix{T}; # d*K
               lr::T=1e-4,    # learning rate
         maxiters::Int=2000,  # maximum iterations
           minerr::Real=1e-4, # abs distance difference below it then stop
          verbose::Bool=false) where T
    γ *= - one(T)
    Kₓₓ = exp.(γ .* pairwise(SqEuclidean(), x, dims=2)) # const
    αᵀKₓₓα = first(α'*Kₓₓ*α)                             # const
    distance = typemax(T)
    trigger = 0

    # ──── prepare backprop variable ────
    z̃ = Info(z, keepsgrad=true, type=Matrix{T})
    xinfos = Vector{XInfo}()
    push!(xinfos, ('w',z̃))
    p = Adam(xinfos; lr, L1decay=0.002)

    # ──── target: min d² = min‖∑ⱼβⱼ*ϕ(zⱼ) - ∑ᵢαᵢ*ϕ(xᵢ)‖² ────
    for i ∈ 1:maxiters
        Kzz = exp(γ .* sqeuc(   z̃, dims=2)) # auto-grad
        Kxz = exp(γ .* sqeuc(x, z̃, dims=2)) # auto-grad
        Kzx = transpose(ᵈ(Kxz))

        # ──── use analytic solution by ∂J/∂β = 0 ────
        β = λI₊(ᵈ(Kzz), 1e-5) \ (Kzx * α)         # K*1
        # ──── min d² ≡ min(β'*Kzz*β - 2α'*Kxz*β) ────
        δ = β'*Kzz*β - 2α'*Kxz*β

        d = sqrt(first(ᵈ(δ)) + αᵀKₓₓα)
        Δ = abs(distance - d)
        distance = d
        if Δ < minerr
            trigger += 1
        else
            trigger = 0
        end
        verbose && println("$i: centers-distance: ", d)
        Mira.backward(δ)
        Mira.update!(p)
        Mira.zerograds!(p)
        if isequal(i, maxiters) || trigger > 100
            return d, β, ᵈ(z̃)
        end
    end
end



# β z 梯度求解
function rebasezb(γ::T,
                  α::Matrix{T}, # N*1
                  x::Matrix{T}, # d*N
                  z::Matrix{T}; # d*K
                 lr::T=1e-4,
           maxiters::Int=2000,
             minerr::Real=1e-4, # abs distance difference below it then stop
            verbose::Bool=false) where T
    γ *= - one(T)
    Kₓₓ = exp.(γ .* pairwise(SqEuclidean(), x, x, dims=2)) # const
    Kzz = exp.(γ .* pairwise(SqEuclidean(), z, z, dims=2)) # const
    Kzx = exp.(γ .* pairwise(SqEuclidean(), z, x, dims=2)) # const
    αᵀKₓₓα = first(α'*Kₓₓ*α)
    distance = typemax(T)
    trigger = 0
    b = λI₊(Kzz, 1e-5) \ (Kzx * α) # K*1

    # ──── prepare backprop variables ────
    β = Info(b, keepsgrad=true, type=Matrix{T})
    z̃ = Info(z, keepsgrad=true, type=Matrix{T})
    xinfos = Vector{XInfo}()
    push!(xinfos, ('w',z̃))
    push!(xinfos, ('w',β))
    p = Adam(xinfos; lr, L1decay=0.002)
    
    # ──── target: min d² = min‖∑ⱼβⱼ*ϕ(zⱼ) - ∑ᵢαᵢ*ϕ(xᵢ)‖² ────
    for i ∈ 1:maxiters
        Kzz = exp(γ .* sqeuc(   z̃, dims=2)) # auto-grad
        Kxz = exp(γ .* sqeuc(x, z̃, dims=2)) # auto-grad
        δ = β'*Kzz*β - 2α'*Kxz*β

        d = sqrt(first(ᵈ(δ)) + αᵀKₓₓα)
        Δ = abs(distance - d)
        distance = d
        if Δ < minerr
            trigger += 1
        else
            trigger = 0
        end
        verbose && println("$i: centers-distance: ",d)

        Mira.backward(δ)
        Mira.update!(p)
        Mira.zerograds!(p)

        if isequal(i, maxiters) || trigger > 100
            return d, ᵈ(β), ᵈ(z̃)
        end
    end
end



# β & z 求解
function rebase(model::SVDD{T,N},
             features::Matrix{T},
                    K::Int,
                    γ::Real;
                   lr::Real=1e-4,
               niters::Int=5,
             maxiters::Int=2000,
               minerr::Real=1e-3,
       beta_uses_grad::Bool=true,
              verbose::Bool=false) where {T,N}
    # ──── select K init bases z from K medoids of features ────
    𝕜 = kernelf(model)
    Δ = distmat(𝕜, features)
    c₁, L₁ = kcentreids(Δ, K; niters, verbose)
    c₂, L₂ = kcentreids(Δ, K; niters, verbose)
    cids = L₁ < L₂ ? c₁ : c₂
    z = features[:, cids] # chose low cost
    
    # ──── update base z and its coefficients β ────
    α = alphas(model)   # N*1
    x = svs(model)      # d*N
    d, β, z = if beta_uses_grad
        rebasezb(T(γ), α, x, z; lr, maxiters, minerr)
    else
        rebase(T(γ), α, x, z; lr, maxiters, minerr)
    end
    println("═══ centers distance: $d ═══")

    # ──── re-estimate R² and <cᵩ, cᵩ> ────
    b   = onesv(model)
    Kbb = 𝕜(b, b) # 1*1
    Kzb = 𝕜(z, b) # K*1
    βKβ = β' * 𝕜(z,z) * β
    R²  = Kbb + βKβ - 2β' * Kzb

    return SVDD{T,N}(first(R²), first(βKβ), reshape(β,1,:), z, 𝕜)
end


