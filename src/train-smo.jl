"""
    smosvdd(kernel::Function,
                 x::Matrix{<:AbstractFloat},
                 C::Real,
                 ϵ::Real=1e-3;
          maxiters::Int=1000,
        checkcycle::Int=10,
       updatecycle::Int=25,
           verbose::Bool=false)

+ Real symmetric square matrix `K = kernel(x,x) ∈ Rⁿⁿ` shall be a Positive Semidefinite Matrix theoretically, i.e.
    `∀ α ∈ Rⁿ`, `K` satisfies `αᵀKα ≥ 0`.
+ `x` is the feature data with `n`-columns
+ `C` is the penalty coefficient (the bigger the less error allowed),
+ lagrange multipliers less than `ϵ` will be discarded.
+ `maxiters` is usually multiple times of `n`, like 3~10, depends on `x`
+ every `checkcycle` times, check if it's converged
+ every `updatecycle` times, update radius once, and maybe print training info if `verbose`.
+ if `verbose`, print training informations.
!!! warn
    If `K` is NOT a Positive Semidefinite Matrix, then the dual problem is no longer a convex optimization,
    but a local optimum, which theoretically loses the global optimum guarantee. Worst converged steps is `O(n²)`
"""
function smosvdd(kernel::Function,
                 x::Matrix{T},
                 C::Real,
                 ϵ::Real=1e-3;
              updatecycle::Int=25,
               checkcycle::Int=10,
                 maxiters::Int=1000,
                  verbose::Bool=false) where T <: AbstractFloat
    N = size(x, 2)
    K = kernel(x, x)
    KROWS, KCOLS = size(K)
    @assert KROWS == KCOLS == N "dimensions dismatch"
    N⁻¹ = T(inv(N))
    if C < N⁻¹
        BUG = """
        due to the constraints:
            ∑ᵢαᵢ= 1
            0 ≤ αᵢ ≤ C
            i = 1 ... N
        the penalty coefficient C s.t. C ≥ 1 / N, but got $C ≥ 1 / $N .
        Now C would be set as 1 to make sure the code runs smoothly.
        """
        @warn BUG
        C = one(T)
    end
    # 增量连续几次不变则终止迭代
    oldΔQα = zero(T)
    triger = zero(Int)
    ONE₋ = T(0.99)
    ONE₊ = T(1.01)
    ε = T(1e-11)  # to avoid too small η = Kᵢᵢ + Kⱼⱼ - 𝟐*Kᵢⱼ and (ε + 0)/(ε + 0)
    ϵ = T(ϵ)      # ignore too small lagrange multipliers

    C = T(C)
    𝟐 = T(2)
    O = zero(T)
    α = fill!(Matrix{T}(undef, N,1), N⁻¹)
    αᵀ = reshape(α, N) # vector
    Kd = diag(K)       # vector
    R² = one(T)

    # 计算初始目标函数值 Q(α) = αᵀKα - diag(K)α
    αᵀKα = first(α' * K * α)
    Qα = αᵀKα - sum(Kd .* αᵀ)
    Δg = Vector{T}(undef, N)
    g  = reshape(K * α, N)
    D² = Kd - 𝟐 .* g .+ αᵀKα
    for t = 1 : maxiters
        # ╭─────────────── 计算违反KKT条件的程度 ───────────────╮
        #= 样本位置与乘子正确搭配就不违反 KKT 条件，对应 XKKT 为 0
        否则违反，且越远离边界越违反, 即对应的 XKKT 值就越大 =#
        E = D² .- R²    # 偏离半径的程度以及方向
        XKKT = abs.(E)  # 偏离半径的绝对程度
        @inbounds @simd for i = 1:N
            D²ᵢ = D²[i]
            αᵀᵢ = αᵀ[i]
            if (D²ᵢ < R² && αᵀᵢ ≈ O) ||  # 球内且乘子为 0
               (D²ᵢ > R² && αᵀᵢ ≈ C) ||  # 球外支撑向量
               (D²ᵢ ≈ R² && O<αᵀᵢ<C)     # 球面支撑向量
                XKKT[i] = O
            end
        end

        # ╭──────────────────────────────────────────────────────────────╮
        # 如果 αi 与 αj 随机配对，那么后续的优化过程就是完全可以基于此配对做并行
        idXKKT = findall(XKKT .> O)
        if isempty(idXKKT)
            verbose && println("converged at $t-th iter")
            break
        end
        i = idXKKT[argmax(XKKT[idXKKT])]
        j = 0
        MAX = typemin(T)
        for k ∈ idXKKT
            Δ = abs(E[k] - E[i])
            if Δ > MAX # 启发式配对，选择让目标函数变化最大的
                j = k
                MAX = Δ
            end
        end
        if isequal(i,j)
            verbose && println("only one violating sample, converged at $t-th iter")
            break
        end

        # ╭────────────────── 优化 αᵢ 和 αⱼ ───────────────────────────────╮
        αᵢᵗ = α[i]
        αⱼᵗ = α[j]
        Kᵢᵢ = K[i,i]
        Kⱼⱼ = K[j,j]
        Kᵢⱼ = K[i,j]
        η = Kᵢᵢ + Kⱼⱼ - 𝟐*Kᵢⱼ  # 距离平方
        η < ε && continue      # 太接近的两个样本不考虑
        V = αᵢᵗ + αⱼᵗ          # αᵢ + αⱼ = 1 - ∑ₖαₖ , k≠i,j
        L = max(V - C, O)
        H = min(C, V)
        # 计算 αᵢ 新值并剪裁, 然后计算 αⱼ 新值并剪裁
        αᵢᵗ⁺¹ = αᵢᵗ + ((Kᵢᵢ - Kⱼⱼ)/𝟐 + g[j] - g[i]) / η
        αᵢᵗ⁺¹ = clamp(αᵢᵗ⁺¹,     L, H)
        αⱼᵗ⁺¹ = clamp(V - αᵢᵗ⁺¹, O, C)
        Δαᵢ = αᵢᵗ⁺¹ - αᵢᵗ
        Δαⱼ = αⱼᵗ⁺¹ - αⱼᵗ

        # 更新 αᵢ αⱼ
        α[i] = αᵢᵗ⁺¹
        α[j] = αⱼᵗ⁺¹

        # ╭────────────────── 更新目标函数 Qα 与二次项 αᵀKα ───────────────╮
        uᵢ = g[i] - αᵢᵗ * Kᵢᵢ - αⱼᵗ * Kᵢⱼ
        uⱼ = g[j] - αᵢᵗ * Kᵢⱼ - αⱼᵗ * Kⱼⱼ

        ΔT₁ = Δαᵢ * Kᵢᵢ + Δαⱼ * Kⱼⱼ
        ΔT₂ = (αᵢᵗ⁺¹^2 - αᵢᵗ^2) * Kᵢᵢ + (αⱼᵗ⁺¹^2 - αⱼᵗ^2) * Kⱼⱼ
        ΔT₃ = 𝟐 * (αᵢᵗ⁺¹ * αⱼᵗ⁺¹ - αᵢᵗ * αⱼᵗ) * Kᵢⱼ + 𝟐 * Δαᵢ * uᵢ + 𝟐 * Δαⱼ * uⱼ

        ΔαᵀKα = ΔT₂ + ΔT₃
        ΔQα = ΔT₁ - ΔαᵀKα
        αᵀKα += ΔαᵀKα
        Qα += ΔQα

        # ╭────────────────── g 与 D² ───────────────╮
        Δg .= reshape(Δαᵢ .* K[:,i] + Δαⱼ .* K[:,j],N)
        g .+= Δg
        D².+= ΔαᵀKα .- Δg .* 𝟐

        # ╭────────────────── 更新半径 ──────────────────╮
        if mod(t, updatecycle) == 1
            idsv = findall(x-> ϵ < x < C, αᵀ)
            n = length(idsv)
            n > 0 && (R² = sum(D²[idsv]) .* inv(T(n)))
            if isnan(R²)
                R² = one(T)
                verbose && println("met a NaN radius")
            end
            verbose && println("iter=$t, score=$Qα, R²=", R²)
        end

        # ╭────────────────── 早停逻辑，增量连续几次不变 ───────────────╮
        if ONE₋ < (oldΔQα + ε)/(ΔQα + ε) < ONE₊
            triger += 1
            if triger > checkcycle
                verbose && println("⏰ early stop at $t-th iteration")
                break
            end
        else
            triger = 0
            oldΔQα = ΔQα
        end
    end

    # ╭────────────────────────── 提取支撑向量 ──────────────────────────────╮
    # 提取所有球面以及球外支撑向量，因为 αᵢ 一直被困在区间 [0,C], 所以无需检查 ≤ C
    i⁰ᶜ = findall(x -> ϵ < x, α) # all support vectors
    iszero(length(i⁰ᶜ)) && @warn "there is no support vector"

    αᵢ = α[i⁰ᶜ, :]
    Xᵢ = x[:, i⁰ᶜ]
    return SVDD{T,1}(R², αᵀKα, reshape(αᵢ,1,:), Xᵢ, kernel)
end


# baseline, debug only
function rawsmosvdd(kernel::Function,
                 x::Matrix{T},
                 C::Real,
                 ϵ::Real=1e-3;
               checkcycle::Int=10,
                 maxiters::Int=1000,
                  verbose::Bool=false) where T <: AbstractFloat
    N = size(x, 2)
    K = kernel(x, x)
    KROWS, KCOLS = size(K)
    @assert KROWS == KCOLS == N "dimensions dismatch"
    N⁻¹ = T(inv(N))
    if C < N⁻¹
        BUG = """
        due to the constraints:
            ∑ᵢαᵢ= 1
            0 ≤ αᵢ ≤ C
            i = 1 ... N
        the penalty coefficient C s.t. C ≥ 1 / N, but got $C ≥ 1 / $N .
        Now C would be set as 1 to make sure the code runs smoothly.
        """
        @warn BUG
        C = one(T)
    end
    # 增量连续几次不变则终止迭代
    oldΔQα = zero(T)
    triger = zero(Int)
    ONE₋ = T(0.99)
    ONE₊ = T(1.01)

    C = T(C)
    e = T(1e-9)
    𝟐 = T(2)
    O = zero(T)
    α = fill!(Matrix{T}(undef, N,1), N⁻¹)
    αᵀ = reshape(α, N) # vector
    Kd = diag(K)       # vector
    R² = one(T)

    # 计算初始目标函数值 Q(α) = αᵀKα - diag(K)α
    αᵀKα = first(α' * K * α)
    Qα = αᵀKα - sum(Kd .* αᵀ)
    LL  = zeros(T, maxiters)
    for t = 1:maxiters
        s = findall(αᵀ .> 0) # 满足 α > 0 的支持向量
        g  = reshape(K[:,s] * α[s], N) # 每次矩阵计算耗时
        D² = Kd - 𝟐 .* g .+ αᵀKα
        # ╭─────────────── 计算违反KKT条件的程度 ────────────────╮
        #= 样本位置与乘子正确搭配就不违反 KKT 条件，对应 XKKT 为 0
        否则违反，且越远离边界越违反, 即对应的 XKKT 值就越大 =#
        E = D² .- R²
        XKKT = abs.(E)
        @inbounds @simd for i = 1:N
            D²ᵢ = D²[i]
            if (D²ᵢ < R² && αᵀ[i] ≈ O) ||  # 球内且乘子为 0
               (D²ᵢ > R² && αᵀ[i] ≈ C) ||  # 球外支撑向量
               (D²ᵢ ≈ R² && O<αᵀ[i]<C)     # 球面支撑向量
                XKKT[i] = O
            end
        end

        # ╭──────────────────────────────────────────────────────────────╮
        # 如果 αi 与 αj 随机配对，那么后续的优化过程就是完全可以基于此配对做并行
        # 选择 αi, 选择 αj
        idXKKT = findall(XKKT .> O)
        isempty(idXKKT) && (println("converged at $t-th iter");break)
        i = idXKKT[argmax(XKKT[idXKKT])]
        j = 0
        MAX = typemin(T)
        for k in idXKKT
            Δ = abs(E[k] - E[i])
            if Δ > MAX
                j = k
                MAX = Δ
            end
        end
        isequal(i,j) && (println("only one violating sample, converged at $t-th iter");break)
        # ╭────────────────── 优化 αᵢ 和 αⱼ ───────────────────────────────╮
        αᵢᵗ = α[i]
        αⱼᵗ = α[j]
        Kᵢᵢ = K[i, i]
        Kⱼⱼ = K[j, j]
        Kᵢⱼ = K[i, j]
        η = Kᵢᵢ + Kⱼⱼ - 𝟐*Kᵢⱼ  # 距离平方
        η < 1e-12 && continue  # 太接近的两个样本不考虑
        V = αᵢᵗ + αⱼᵗ          # αᵢ + αⱼ = 1 - ∑ₖαₖ , k≠i,j
        L = max(V - C, O)
        H = min(C, V)
        # 计算 αᵢ 新值并剪裁, 然后计算 αⱼ 新值并剪裁
        αᵢᵗ⁺¹ = αᵢᵗ + ((Kᵢᵢ - Kⱼⱼ)/𝟐 + g[j] - g[i])/η
        αᵢᵗ⁺¹ = clamp(αᵢᵗ⁺¹,     L, H)
        αⱼᵗ⁺¹ = clamp(V - αᵢᵗ⁺¹, O, C)
        Δαᵢ = αᵢᵗ⁺¹ - αᵢᵗ
        Δαⱼ = αⱼᵗ⁺¹ - αⱼᵗ

        # 更新 αᵢ αⱼ
        α[i] = αᵢᵗ⁺¹
        α[j] = αⱼᵗ⁺¹

        # ╭────────────────── 更新目标函数 Qα 与二次项 αᵀKα ───────────────╮
        uᵢ = g[i] - αᵢᵗ * Kᵢᵢ - αⱼᵗ * Kᵢⱼ
        uⱼ = g[j] - αᵢᵗ * Kᵢⱼ - αⱼᵗ * Kⱼⱼ

        ΔT₁ = Δαᵢ * Kᵢᵢ + Δαⱼ * Kⱼⱼ
        ΔT₂ = (αᵢᵗ⁺¹^2 - αᵢᵗ^2) * Kᵢᵢ + (αⱼᵗ⁺¹^2 - αⱼᵗ^2) * Kⱼⱼ
        ΔT₃ = 𝟐 * (αᵢᵗ⁺¹ * αⱼᵗ⁺¹ - αᵢᵗ * αⱼᵗ) * Kᵢⱼ + 𝟐 * Δαᵢ * uᵢ + 𝟐 * Δαⱼ * uⱼ

        ΔαᵀKα = ΔT₂ + ΔT₃
        ΔQα = ΔT₁ - ΔαᵀKα
        αᵀKα += ΔαᵀKα
        Qα += ΔQα
        LL[t] = Qα
        # ╭──────────────────── 更新半径 ─────────────────────╮
        Rᵢ = sqrt(Kd[i] + αᵀKα - 𝟐*(g[i] + Δαᵢ*Kᵢᵢ + Δαⱼ*Kᵢⱼ))
        Rⱼ = sqrt(Kd[j] + αᵀKα - 𝟐*(g[j] + Δαⱼ*Kⱼⱼ + Δαᵢ*Kᵢⱼ))
        R² = ((Rᵢ + Rⱼ) / 𝟐)^2
        if isnan(R²)
            R² = one(T)
            verbose && println("met a NaN radius, $ΔQα")
        end
        (mod(t,checkcycle)==0 && verbose) && println("$t-th iter, score=$Qα, R²=", R²)

        # ╭────────────────── 早停逻辑，增量连续几次不变 ───────────────╮
        if ONE₋ < (e + oldΔQα)/(e + ΔQα) < ONE₊
            triger += 1
            if triger > checkcycle
                verbose && println("early stop condition is met at $t-th iteration")
                break
            end
        else
            triger = 0
            oldΔQα = ΔQα
        end
    end

    # ╭──────────────────────── 提取支撑向量 ───────────────────────────╮
    # 提取所有球面以及球外支撑向量，因为 αᵢ 一直被困在区间 [0,C], 所以无需检查 ≤ C
    ϵ = T(ϵ)                     # ignore too small values
    i⁰ᶜ = findall(x -> ϵ < x, α) # all support vectors
    iszero(length(i⁰ᶜ)) && @warn "there is no support vector"

    αᵢ = α[i⁰ᶜ, :]
    Xᵢ = x[:, i⁰ᶜ]
    return SVDD{T,1}(R², αᵀKα, reshape(αᵢ,1,:), Xᵢ, kernel), LL
end
