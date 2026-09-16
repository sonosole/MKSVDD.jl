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


"""
    SVDD(kernel <: XKernel, x::Matrix{Real}, C::Real, ϵ::Real=1e-3; verbose::Bool=false)

`x` is the normal data,  `C` is the penalty coefficient (the bigger the less error allowed), 
lagrange multipliers below the threshold `ϵ` will be discarded.
"""
function SVDD(kernel::KERNEL, x::Matrix{T}, C::Real, ϵ::Real=1e-3; verbose::Bool=false) where {T <: Real, KERNEL <: XKernel}
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

    a = ones(T, N, 1) ./ N
    K = kmat(kernel, x, x, obsdim=2)
    Ki = reshape(diag(K), 1, N);

    # 优化模型参数
    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => verbose));
    @variable(model, a[1:N]);
    @objective(model, Min, sum(a' * K * a) - sum(Ki * a));
    @constraint(model, sum(a) == 1);
    @constraint(model, 0 .≤ a .≤ C);
    status = JuMP.optimize!(model);
    
    α = value.(a)
    ϵ = T(ϵ)
    𝟐 = T(2)
    # 提取支撑向量
    onR = Int[]  # =R s.t. 0 < αᵢ < C
    i0C = Int[]  # ≥R s.t. 0 < αᵢ ≤ C
    for (i, αᵢ) ∈ enumerate(α)
        if ϵ < αᵢ
            if αᵢ < C
                push!(onR, i)
            end
            push!(i0C, i)
        end
    end

    j = 0
    if length(onR) > 0
        j = first(onR)
    else
        @error "there is no support vector"
    end

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


function _SVDD(kernel::KERNEL, x::Matrix{T}, y::Vector{Int}, C::T, ϵ::T=1e-3, verbose::Bool=false) where {T <: Real, KERNEL <: XKernel}
    N = size(x,2)      # number of features
    M = length(y)      # number of labels

    y = reshape(y, N, 1)
    a = ones(T, N, 1) ./ N
    K = T.(y * y') .* kmat(kernel, x, x, obsdim=2)
    Ki = reshape(diag(K), 1, N)

    # 优化模型参数
    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => verbose));
    @variable(model, a[1:N])
    @objective(model, Min, sum(a' * K * a) - sum(Ki * a));
    @constraint(model, sum(y .* a) == 1)
    @constraint(model, 0 .≤ a .≤ C)
    status = JuMP.optimize!(model)
    
    α = value.(a)
    𝟐 = T(2)
    # 提取支撑向量
    onR = Int[]  # =R s.t. 0 < αᵢ < C
    i0C = Int[]  # ≥R s.t. 0 < αᵢ ≤ C
    for (i, αᵢ) ∈ enumerate(α)
        if ϵ < αᵢ
            if αᵢ < C
                push!(onR, i)
            end
            push!(i0C, i)
        end
    end
    αᵢ = α[i0C,:]
    yᵢ = y[i0C,:]
    Wᵢ = yᵢ .* αᵢ
    
    # chose one support vec
    j = 0
    if length(onR) > 0
        j = first(onR)
    else
        @error "there is no support vector"
    end

    Xi = x[:,i0C]
    Xs = x[:,j:j]

    Kss = kmat(kernel, Xs, Xs, obsdim=2)
    Kis = kmat(kernel, Xi, Xs, obsdim=2)
    Kij = kmat(kernel, Xi, Xi, obsdim=2)
    WᵀKW = Wᵢ' * Kij * Wᵢ
    R²   = Kss - 𝟐 * Wᵢ' * Kis + WᵀKW

    return SVDD{T,2}(first(R²), first(WᵀKW), Wᵢ, Xi, kernel)
end


"""
    SVDD(kernel <: XKernel, xpos::Matrix{T}, xneg::Matrix{Real}, C::Real, ϵ::Real=1e-3; verbose::Bool=false)

`xpos` is the normal data and `xneg` is the abnormal data,  `C` is the penalty coefficient (the bigger the less error allowed), 
lagrange multipliers below the threshold `ϵ` will be discarded.
"""
function SVDD(kernel::KERNEL, xpos::Matrix{T}, xneg::Matrix{T}, C::Real, ϵ::Real=1e-3; verbose::Bool=false) where {T <: Real, KERNEL <: XKernel}
    P = size(xpos,2); @assert P > 0 "no positives";
    N = size(xneg,2); @assert N > 0 "no negatives";

    if C < 1 / P
        BUG = """
        due to the constraints:
            ∑ⱼyⱼαⱼ = 1       (origin)
            ∑ₚαₚ - ∑ₙαₙ = 1  (inferred)
            0 ≤ αⱼ ≤ C  ⇒  1 < ∑ₚαₚ ≤ C*P ⇒  C ≥ 1 / P
        where  p ∈ {j | yⱼ = +1, j = 1 ... N}, P = |p|
               n ∈ {j | yⱼ = -1, j = 1 ... N}
        the penalty coefficient C s.t. C ≥ 1 / P, but got $C ≥ 1 / $P .
        Now C would be set as 1 to make sure the code runs smoothly.
        """
        @warn BUG
        C = one(T)
    end

    y = svddlabel(P, N)
    x = hcat(xpos, xneg)
    return _SVDD(kernel, x, y, T(C), T(ϵ), verbose)
end


"""
    SVDD(kernel <: XKernel, x::Matrix{Real}, y::Vector{Int}, C::Real, ϵ::Real=1e-3; verbose::Bool=false)

`x` is the  data with label `y`,  `C` is the penalty coefficient (the bigger the less error allowed), 
lagrange multipliers below the threshold `ϵ` will be discarded. Note that positive samples are labeled 
with +1, while the negative samples are labeled with -1. The function:

```julia
svddlabel(num_of_pos::Int, num_of_neg::Int)::Vector{Int}
```
could be a helper to create labels.
"""
function SVDD(kernel::KERNEL, x::Matrix{T}, y::Vector{Int}, C::Real, ϵ::Real=1e-3; verbose::Bool=false) where {T <: Real, KERNEL <: XKernel}
    L = size(x, 2)  # number of features
    M = length(y)   # number of labels
    @assert L > 0 "no features";
    @assert M > 0 "no labels";
    @assert L == M "number of features ($L) ≠ number of labels($M)"

    P = 0
    for v ∈ y
        isone(v) && (P += 1)
    end
    N = L - P

    @assert P > 0 "no positives";
    @assert N > 0 "no negatives";

    if C < 1 / P
        BUG = """
        due to the constraints:
            ∑ⱼyⱼαⱼ = 1       (origin)
            ∑ₚαₚ - ∑ₙαₙ = 1  (inferred)
            0 ≤ αⱼ ≤ C  ⇒  1 < ∑ₚαₚ ≤ C*P ⇒  C ≥ 1 / P
        where  p ∈ {j | yⱼ = +1, j = 1 ... N}, P = |p|
               n ∈ {j | yⱼ = -1, j = 1 ... N}
        the penalty coefficient C s.t. C ≥ 1 / P, but got $C ≥ 1 / $P .
        Now C would be set as 1 to make sure the code runs smoothly.
        """
        @warn BUG
        C = one(T)
    end

    return _SVDD(kernel, x, y, T(C), T(ϵ), verbose)
end


# inference functor
function (Model::SVDD{T})(feat::Matrix{T}) where T
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
        Ksx = kmat(κ, xs, x, obsdim=2)
        Δ²[i] = first(Kxx - 𝟐W' * Ksx) + WᵀKW
    end
    # Δ² .> R² ⇒ out of sphere
    # Δ² .≡ R² ⇒ on sphere surface
    # Δ² .< R² ⇒ inside sphere
    return Δ² .- R²
end


"""
    svddlabel(num_of_pos::Int, num_of_neg::Int) -> y::Vector{Int}

a helper to create labels. Note that positive samples are labeled with +1, 
while the negative samples are labeled with -1.
"""
function svddlabel(p::Int, n::Int)
    N = p + n
    label = Vector{Int}(undef, N)
    label[1   : p] .= +1
    label[p+1 : N] .= -1
    return label
end

