mutable struct SVDD{T, N, K <: XKernel}
    R²     :: T
    WᵀKW   :: T
    𝟐W     :: Matrix{T}
    svecs  :: Matrix{T}
    kernel :: K
    function SVDD{T,N}(R²::Real, WᵀKW::Real, W::Matrix, svecs::Matrix, kernel::K) where {T, N, K <: XKernel}
        @assert N==1 || N==2 begin
            "only support type SVDD{T,1} or SVDD{T,2}, got SVDD{T,$N}"
        end
        new{T,N,K}(R², WᵀKW, 2W, svecs, kernel)
    end
end


@inline nsvs(S::SVDD) = length(S.𝟐W)


function Base.show(io::IO, ::MIME"text/plain", svdd::SVDD{T,N,K}) where {T, N, K <: XKernel}
    C = nsvs(svdd)
    print(io, "SVDD{$T,$N,$K} with $C support vectors")
end


function SVDD(kernel::KERNEL, x::Matrix{T}, C::Real, ϵ::Real=1e-3) where {T <: Real, KERNEL <: XKernel}
    C = T(C)
    N = size(x, 2)
    @assert C ≥ 1 / N  begin
        """
        due to the constraints:
            ∑ᵢ αᵢ = 1
            0 ≤ αᵢ ≤ C
            i = 1 ... N
        the penalty coefficient C s.t. C ≥ 1 / N, but got $C ≥ 1 / $N
        """
    end
    a = ones(T, N, 1) ./ N
    K = kmat(kernel, x, x, obsdim=2)
    Ki = reshape(diag(K), 1, N);

    # 优化模型参数
    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => true));
    @variable(model, a[1:N]);
    @objective(model, Min, sum(a' * K * a) - sum(Ki * a));
    @constraint(model, sum(a) == 1);
    @constraint(model, 0 .≤ a .≤ C);
    status = JuMP.optimize!(model);
    
    α = value.(a)
    ϵ = T(ϵ)
    𝟐 = T(2)
    # 提取支撑向量
    idS = Int[]  # =R s.t. 0 < αᵢ < C
    i0C = Int[]  # ≥R s.t. 0 < αᵢ ≤ C
    for (i, αᵢ) ∈ enumerate(α)
        if ϵ < αᵢ
            if αᵢ < C - ϵ
                push!(idS, i)
            end
            push!(i0C, i)
        end
    end
    j  = idS[1]
    αᵢ = α[i0C,:]
    Xi = x[:,i0C]
    Xs = x[:,j:j]

    Kss = kmat(kernel, Xs, Xs, obsdim=2)
    Kis = kmat(kernel, Xi, Xs, obsdim=2)
    Kij = kmat(kernel, Xi, Xi, obsdim=2)
    αᵀKα = αᵢ' * Kij * αᵢ
    R²   = Kss - 𝟐 * αᵢ' * Kis + αᵀKα

    return SVDD{T,1}(first(R²), first(αᵀKα), αᵢ, Xi, kernel)
end


function SVDD(kernel::KERNEL, x::Matrix{T}, y::Vector{Int}, C::Real, ϵ::Real=1e-3) where {T <: Real, KERNEL <: XKernel}
    C = T(C)           # constraint 0 ≤ αᵢ ≤ C
    N = size(x, 2)     # number of features
    M = length(y)      # number of labels
    @assert M == N "number of features ($N) ≠ number of labels($M)"
    @assert C > 1 / N  # ∑ᵢ(yᵢ * αᵢ) = 1

    y = reshape(y, N, 1)
    a = ones(T, N, 1) ./ N
    K = T.(y * y') .* kmat(kernel, x, x, obsdim=2)
    Ki = reshape(diag(K), 1, N)

    # 优化模型参数
    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => true));
    @variable(model, a[1:N])
    @objective(model, Min, sum(a' * K * a) - sum(Ki * a));
    @constraint(model, sum(y .* a) == 1)
    @constraint(model, 0 .≤ a .≤ C)
    status = JuMP.optimize!(model)
    
    α = value.(a)
    ϵ = T(ϵ)
    𝟐 = T(2)
    # 提取支撑向量
    idS = Int[]  # supports s.t. 0 < αᵢ < C
    i0C = Int[]  # outers s.t.   0 < αᵢ ≤ C
    for (i, αᵢ) ∈ enumerate(α)
        if ϵ < αᵢ
            if αᵢ < C - ϵ
                push!(idS, i)
            end
            push!(i0C, i)
        end
    end
    αᵢ = α[i0C,:]
    yᵢ = y[i0C,:]
    Wᵢ = yᵢ .* αᵢ
    j  = idS[1]   # chose one support vec
    Ys = y[j,:]
    Xi = x[:,i0C]
    Xs = x[:,j:j]

    Kss = kmat(kernel, Xs, Xs, obsdim=2)
    Kis = kmat(kernel, Xi, Xs, obsdim=2)
    Kij = kmat(kernel, Xi, Xi, obsdim=2)
    WᵀKW = Wᵢ' * Kij * Wᵢ
    R²   = Kss - 𝟐 * Wᵢ' * Kis + WᵀKW

    return SVDD{T,2}(first(R²), first(WᵀKW), Wᵢ, Xi, kernel)
end


function SVDD(kernel::KERNEL, xpos::Matrix{T}, xneg::Matrix{T}, C::Real, ϵ::Real=1e-3) where {T <: Real, KERNEL <: XKernel}
    x = hcat(xpos, xneg)
    y = svddlabel(size(xpos,2), size(xneg,2))
    return SVDD(kernel, x, y, C, ϵ)
end


function (Model::SVDD)(feat::Matrix{T}) where T
    WᵀKW = Model.WᵀKW
    xs   = Model.svecs
    R²   = Model.R²
    𝟐W   = Model.𝟐W
    κ    = Model.kernel
    N  = size(feat, 2)
    Δ² = Vector{T}(undef, N)
    for i ∈ 1:N
        x = feat[:, i:i]
        Kxx = kmat(κ, x,  x, obsdim=2)
        Kis = kmat(κ, xs, x, obsdim=2)
        Δ²[i] = first(Kxx - 𝟐W' * Kis .+ WᵀKW) - R²
    end
    return Δ²
end


function svddlabel(p::Int, n::Int)
    N = p + n
    label = Vector{Int}(undef, N)
    label[1   : p] .= +1
    label[p+1 : N] .= -1
    return label
end

