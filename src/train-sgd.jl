using Mira

"""
         sgdsvdd(x::Matrix{T},
                 C::Real,
                 ϵ::Real=1e-3;
                lr::Real=1e-3,
                L1::Real=1e-3,
        checkcycle::Int=50,
          maxiters::Int=1000,
            minerr::Real=1e-4, # abs distance difference below it then stop
           verbose::Bool=false) where T

Gradient based SVDD training method.

+ `x` is the feature data with `n`-columns
+ `C` is the penalty coefficient (the bigger the less error allowed),
+ lagrange multipliers less than `ϵ` will be discarded.
+ `lr` is learning rate
+ `L1` is L1 coefficient of regularity
+ `maxiters` is usually multiple times of `n`, like 3~10, depends on `x`
+ every `checkcycle` times, check if it's converged
+ if `verbose`, print training informations.
"""
function sgdsvdd(x::Matrix{T},
                 C::Real,
                 ϵ::Real=1e-3;
                lr::Real=1e-3,
                L1::Real=1e-3,
        checkcycle::Int=50,
          maxiters::Int=1000,
            minerr::Real=1e-4, # abs distance difference below it then stop
           verbose::Bool=false) where T
    featdims, N = size(x)
    a = Info(T(1e-2) .* randn(T,N,1), keepsgrad=true, type=Matrix{T})
    r = Info(T(1e-2) .* randn(T,1,1), keepsgrad=true, type=Matrix{T})
    xinfos = Vector{XInfo}()
    push!(xinfos, ('w',a))
    push!(xinfos, ('w',r))
    p = Adam(xinfos; lr, L1decay=L1)

    trigger = 0
    COST = typemax(T)
    dₓₓ = pairwise(SqEuclidean(), x, x; dims=2)
    for iter = 1:maxiters
        γ = - r^2         # so γ is always negative
        K = exp(γ .* dₓₓ) # RBF kernel
        a² = a^2          # so α is always positive
        α  = a² .* inv(sum(a²,dims=(1,2)))
        L = α'*K*α - α' * cdiag(K) + sum(α ≤ C, dims=(1,2))

        LOSS = cost(L)
        Δ = abs(COST - LOSS)
        COST = LOSS
        if Δ < minerr
            trigger += 1
        else
            trigger = 0
        end

        verbose && println("$iter loss: ", LOSS)

        Mira.backward(L)
        Mira.update!(p)
        Mira.zerograds!(p)

        if isequal(iter, maxiters) || trigger > checkcycle
            γ  = first(- ᵈ(r) .^ 2)
            a² = ᵈ(a) .^ 2
            println("╠════ the coeff γ in exp(-γ‖x-y‖²) is $(-γ) ════╣")
            α = vec(a² .* inv(sum(a²,dims=(1,2))))
            s = findfirst(g-> ϵ < g < C, α) # support vector on sphere surface
            isnothing(s) && error("there is no support vector")
            i = findall(g -> g > ϵ, α)      # all support vectors
            ker(X::AbstractMatrix, Y::AbstractMatrix) = exp.(γ .* pairwise(SqEuclidean(), X, Y, dims=2))
            αᵢ = reshape(α[i], 1, :)
            Xᵢ = x[:, i]
            Xs = x[:,s:s]
            Kss = ker(Xs, Xs)
            Kis = ker(Xᵢ, Xs)
            Kij = ker(Xᵢ, Xᵢ)
            αᵀKα = αᵢ * Kij * αᵢ'
            R²   = Kss - 2αᵢ*Kis + αᵀKα
            return SVDD{T,1}(first(R²), first(αᵀKα), αᵢ, Xᵢ, ker), γ
        end
    end
end

