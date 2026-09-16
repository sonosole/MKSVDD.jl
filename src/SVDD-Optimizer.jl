"""
    svdd(kernel::Function, x::Matrix{<:AbstractFloat}, C::Real, ϵ::Real=1e-3; verbose::Bool=false)

+ `x` is the normal feature data
+ `C` is the penalty coefficient (the bigger the less error allowed),
+ lagrange multipliers below `ϵ` will be discarded.
+ if `verbose`, print out the training informations
"""
function svdd(kernel::Function, x::Matrix{T}, C::Real, ϵ::Real=1e-3; verbose::Bool=false) where {T <: AbstractFloat}
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

    a = fill!(Matrix{T}(undef, N,1), T(inv(N)))
    K = kernel(x, x)
    Ki = reshape(diag(K), 1, N);

    # 优化模型参数
    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => verbose));
    @variable(model, a[1:N]);
    @objective(model, Min, sum(a' * K * a) - sum(Ki * a));
    @constraint(model, sum(a) == 1);
    @constraint(model, 0 .≤ a .≤ C);
    status = JuMP.optimize!(model);

    α = value.(a)
    ϵ = T(abs(ϵ))
    𝟐 = T(2f0)
    # ╭────────────────────────── 提取支撑向量 ──────────────────────────────╮
    # 提取所有球面以及球外支撑向量，因为 αᵢ 一直被困在区间 [0,C], 所以无需检查 ≤ C
    i0C = findall(x-> ϵ < x, α)
    # 只提取在球面的一个支撑向量，必须检查小于C
    s = findfirst(x-> ϵ < x < C, α)
    isnothing(s) && error("there is no support vector")

    αᵢ = α[i0C,:]
    Xi = x[:,i0C]
    Xs = x[:,s:s]

    Kss = kernel(Xs, Xs)
    Kis = kernel(Xi, Xs)
    Kij = kernel(Xi, Xi)
    αᵀKα = αᵢ' * Kij * αᵢ
    R²   = Kss - 𝟐 * αᵢ' * Kis + αᵀKα

    return SVDD{T,1}(first(R²), first(αᵀKα), αᵢ, Xi, kernel)
end


function _svdd(kernel::Function, x::Matrix{T}, y::Vector{Int}, C::T, ϵ::T=1e-3, verbose::Bool=false) where {T <: AbstractFloat}
    N = size(x,2)      # number of features
    M = length(y)      # number of labels

    y = reshape(y, N, 1)
    a = fill!(Matrix{T}(undef, N,1), T(inv(N)))
    K = T.(y * y') .* kernel(x, x)
    Ki = reshape(diag(K), 1, N)

    # 优化模型参数
    model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => verbose));
    @variable(model, a[1:N])
    @objective(model, Min, sum(a' * K * a) - sum(Ki * a));
    @constraint(model, sum(y .* a) == 1)
    @constraint(model, 0 .≤ a .≤ C)
    status = JuMP.optimize!(model)

    ϵ = T(abs(ϵ))
    α = value.(a)
    𝟐 = T(2)
    # ╭────────────────────────── 提取支撑向量 ──────────────────────────────╮
    # 提取所有球面以及球外支撑向量，因为 αᵢ 一直被困在区间 [0,C], 所以无需检查 ≤ C
    i0C = findall(x-> ϵ < x, α)
    # 只提取在球面的一个支撑向量，必须检查小于C
    s = findfirst(x-> ϵ < x < C, α)
    isnothing(s) && error("there is no support vector")

    αᵢ = α[i0C,:]
    yᵢ = y[i0C,:]
    wᵢ = yᵢ .* αᵢ
    Xi = x[:,i0C]
    Xs = x[:,s:s]

    Kss = kernel(Xs, Xs)
    Kis = kernel(Xi, Xs)
    Kij = kernel(Xi, Xi)
    wᵀKw = wᵢ' * Kij * wᵢ
    R²   = Kss - 𝟐 * wᵢ' * Kis + wᵀKw

    return SVDD{T,2}(first(R²), first(wᵀKw), wᵢ, Xi, kernel)
end


"""
    svdd(kernel::Function, xpos::Matrix{T}, xneg::Matrix{T}, C::Real, ϵ::Real=1e-3; verbose::Bool=false) where {T <: AbstractFloat}

+ `xpos` is the normal data
+ `xneg` is the abnormal data
+ `C` is the penalty coefficient (the bigger the less error allowed)
+ lagrange multipliers less than `ϵ` will be discarded.
"""
function svdd(kernel::Function, xpos::Matrix{T}, xneg::Matrix{T}, C::Real, ϵ::Real=1e-3; verbose::Bool=false) where {T <: AbstractFloat}
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
    return _svdd(kernel, x, y, T(C), T(ϵ), verbose)
end


"""
    svdd(kernel::Function, x::Matrix{T}, y::Vector{Int}, C::Real, ϵ::Real=1e-3; verbose::Bool=false) where {T <: AbstractFloat}

`x` is the data with label `y`,  `C` is the penalty coefficient (the bigger the less error allowed),
lagrange multipliers less than `ϵ` will be discarded. Note that positive samples are labeled
with +1, while the negative samples are labeled with -1. The function:

```julia
svddlabel(num_of_pos::Int, num_of_neg::Int)::Vector{Int}
```
could be a helper to create labels.
"""
function svdd(kernel::Function, x::Matrix{T}, y::Vector{Int}, C::Real, ϵ::Real=1e-3; verbose::Bool=false) where {T <: AbstractFloat}
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

    return _svdd(kernel, x, y, T(C), T(ϵ), verbose)
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
