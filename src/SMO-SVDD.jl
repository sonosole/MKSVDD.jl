function SMOSVDD(kernel::KERNEL, x::Matrix{T}, C::Real, ϵ::Real=1e-3; maxiters::Int = 1000) where {T <: Real, KERNEL <: MKSVDD.XKernel}
    C = T(C)
    N = size(x, 2)
    if C < 1 / N
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
    e = T(1e-9)
    𝟐 = T(2)
    O = zero(T)
    α = ones(T, N, 1) ./ N
    K = kmat(kernel, x, x, obsdim=2)
    Kd = reshape(diag(K), 1, N);
    αᵀ = reshape(α, N)
    R² = one(T)

    # 计算初始目标函数值
    αᵀKα = first(α' * K * α)
    Qα = αᵀKα - first(Kd * α)
    
    # 增量连续几次不变则终止迭代
    oldΔQα = zero(T)
    triger = 0
    ONE₋ = T(0.99)
    ONE₊ = T(1.01)

    for t = 1:maxiters
        idSV = findall(αᵀ .> 0)
        g = α[idSV]' * K[idSV, :]
        D² = reshape(abs.(Kd .- 𝟐 .* g .+ αᵀKα), N)
        
        #= 计算违反KKT条件的程度，
        样本位置与乘子正确搭配的就不违反条件，
        否则违反，且越远离边界越违反
        =#
        XKKT = abs.(D² .- R²)
        @inbounds @simd for i = 1:N
            if (D²[i] < R² && αᵀ[i] == O) ||  # 乘子为零则应该在内部
               (D²[i] ≥ R² && αᵀ[i] == C) ||  # 乘子见顶且位于表面或外部
               (D²[i] == R² && O < αᵀ[i] < C)
                XKKT[i] = O
            end
        end

        # ╭─────────────────────────────────────────────────────────────────╮
        # 如果 αi 与 αj 随机配对，那么后续的优化过程就是完全可以基于此配对做并行
        # 选择 αi
        idXKKT = findall(XKKT .> O)        # 选择违反的
        idNonBound = findall(0 .< αᵀ .< C) # 非边界支持向量
        idNonBoundXKKT = intersect(idXKKT, idNonBound)
        if length(idNonBoundXKKT) > 0
            maxidx = argmax(XKKT[idNonBoundXKKT])
            i = idNonBoundXKKT[maxidx]
        else
            i = argmax(XKKT)
        end

        #  选择 αj
        j = idXKKT[rand(1:length(idXKKT))]
        while isequal(i,j) # 避免相同样本
            j = rand(1:N)
        end
        # ╰─────────────────────────────────────────────────────────────────╯

        # ╭────────────────── 优化 αᵢ 和 αⱼ ───────────────────────────────╮
        αᵢᵗ = α[i]
        αⱼᵗ = α[j]
        Kᵢᵢ = K[i,i]
        Kⱼⱼ = K[j,j]
        Kᵢⱼ = K[i,j]
        V = αᵢᵗ + αⱼᵗ          # αᵢ + αⱼ = 1 - ∑ₖαₖ , k≠i,j
        η = Kᵢᵢ + Kⱼⱼ - 𝟐*Kᵢⱼ  # 距离平方
        L = max(V - C, O)
        H = min(C, V)
        # 计算 αᵢ 新值并剪裁, 然后计算 αⱼ 新值并剪裁
        αᵢᵗ⁺¹ = αᵢᵗ + ((Kᵢᵢ - Kⱼⱼ)/2 + g[j] - g[i])/η
        αᵢᵗ⁺¹ = clamp(αᵢᵗ⁺¹, L, H)
        αⱼᵗ⁺¹ = clamp(V - αᵢᵗ⁺¹, O, C)
        
        # 更新 αᵢ αⱼ
        α[i] = αᵢᵗ⁺¹
        α[j] = αⱼᵗ⁺¹

        # ╭────────────────── 更新目标函数 Qα 与二次项 αᵀKα ───────────────╮
        uᵢ = g[i] - αᵢᵗ * Kᵢᵢ - αⱼᵗ * Kᵢⱼ
        uⱼ = g[j] - αᵢᵗ * Kᵢⱼ - αⱼᵗ * Kⱼⱼ
        Δαᵢ = αᵢᵗ⁺¹ - αᵢᵗ
        Δαⱼ = αⱼᵗ⁺¹ - αⱼᵗ
        ΔT₁ = Δαᵢ * Kᵢᵢ + Δαⱼ * Kⱼⱼ
        ΔT₂ = (αᵢᵗ⁺¹^2 - αᵢᵗ^2) * Kᵢᵢ + (αⱼᵗ⁺¹^2 - αⱼᵗ^2) * Kⱼⱼ
        ΔT₃ = 𝟐*(αᵢᵗ⁺¹ * αⱼᵗ⁺¹ - αᵢᵗ * αⱼᵗ) * Kᵢⱼ + 𝟐*Δαᵢ * uᵢ + 𝟐*Δαⱼ * uⱼ

        ΔαᵀKα = ΔT₂ + ΔT₃
        ΔQα   = ΔT₁ - ΔαᵀKα
        αᵀKα += ΔαᵀKα
        Qα   += ΔQα

        # ╭────────────────── 早停逻辑，增量连续几次不变 ───────────────╮
        if ONE₋ < (e + oldΔQα)/(e + ΔQα) < ONE₊
            triger += 1
            if triger > 5
                println("iter stops at $t")
                break
            end
        else
            triger = 0
        end
        oldΔQα = ΔQα

        # ╭────────────────── 更新半径 ───────────────╮
        if 0 < α[i] < C
            R²  = abs(Kd[i] + αᵀKα - 𝟐*(g[i] + Δαᵢ*Kᵢᵢ + Δαⱼ*Kᵢⱼ))
        elseif 0 < α[j] < C
            R²  = abs(Kd[j] + αᵀKα - 𝟐*(g[j] + Δαᵢ*Kᵢⱼ + Δαⱼ*Kⱼⱼ))
        else
            R²ᵢ = abs(Kd[i] + αᵀKα - 𝟐*(g[i] + Δαᵢ*Kᵢᵢ + Δαⱼ*Kᵢⱼ))
            R²ⱼ = abs(Kd[j] + αᵀKα - 𝟐*(g[j] + Δαᵢ*Kᵢⱼ + Δαⱼ*Kⱼⱼ))
            R²  = (R²ᵢ + R²ⱼ) / 2
        end
        if isnan(R²)
            R² = one(T)
        end
    end
    
    # ╭────────────────── 提取支撑向量 ───────────────╮
    ϵ = T(ϵ)     # ignore too small values
    ⁰iᶜ = Int[]  # ≥R s.t. ϵ < αᵢ ≤ C
    for (i, αᵢ) ∈ enumerate(α)
        ϵ < αᵢ && push!(⁰iᶜ, i)
    end
    iszero(length(⁰iᶜ)) && @warn "there is no support vector"

    αᵢ = α[⁰iᶜ,:]
    Xᵢ = x[:,⁰iᶜ]

    return SVDD{T,1}(R², αᵀKα, αᵢ, Xᵢ, kernel)
end


