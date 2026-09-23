"""
    distmat(kernel::Function, x::AbstractArray) -> Δ
Return squre shaped distance matrix by `kernel`.

    Δ[i,j] = K[i,i] + K[j,j] - 2K[i,j] where K[i,j] = kernel(xᵢ, xⱼ)
"""
function distmat(kernel::Function, x::AbstractArray)
    K = kernel(x, x)
    N = size(K, 1)
    Δ = similar(x, N, N)
    d = diag(K)
    for i = 1:N
        for j = 1:i
            Δ[i,j] = d[i] + d[j] - 2K[i,j]
            Δ[j,i] = Δ[i,j]
        end
    end
    return Δ
end


"""
    kcentreids(Δ::Matrix{T},
               K::Int;
          niters::Int=5,
         verbose::Bool=false) -> ids_of_K_centers::Vector{Int}, LOSS::T
Return indexes of `K` centers according to distance matrix `Δ` and cluster `LOSS`
+ `niters` is the number of iterations
+ if `verbose`, print the k-medoids loss
"""
function kcentreids(Δ::Matrix{T}, K::Int; niters::Int=5, verbose::Bool=false) where T
    N, M = size(Δ);
    @assert N == M "Distance Matrix shall be squred, but got $N*$M"
    @assert N ≥ K  "No enough data to train"
    if isequal(N, K)
        return collect(1:N), zero(T)
    end
    c = shuffle(1:N)[1:K]    # choose K svs as init start
    t = 0
    L = zero(T)
    verbose && println("─────── iter $K-medoids by distance matrix ────────")
    while t < niters
        t = t + 1
        d = Δ[c,:] # distances to center c, size K*N
        # ╭────────────── e-step ──────────────────╮
        valmin, idxmin = findmin(d, dims=1)
        L = sum(valmin)
        verbose && println("iter $t, cluster-loss=$L")
        # record which center it belongs to
        cids = vec(@. first(Tuple(idxmin)))
        # ╭──────────── m-step ───────────────╮
        for k ∈ 1:K
            kidxs = findall(u->u==k, cids) # idx belong to cluster k, i.e. [1,3,9,...,maxid≤N]
            kdist = Δ[kidxs, kidxs]        # distance matrix inside cluster k
            # ╭─── 在 k 簇内以各个点为中心，所有点到各个中心的距离之和 ───╮
            # ╰───── 将距离之和最小的那个点作为 k 簇的新中心 ────────────╯
            sumdist = sum(kdist, dims=1)   # total distance to each point inside cluster k
            cidbest = argmin(vec(sumdist)) # note: it's local index, have to be converted to global index
            c[k] = kidxs[cidbest]
        end
    end
    return c, L
end


"""
    kcentres(Δ::Matrix{T},
             K::Int;
        niters::Int=5,
       verbose::Bool=false) -> ids_of_K_centers::Vector{Int}, vecids_of_K_centers::Vector{Vector{Int}}
Return indexes of `K` centers and its coresponding samples indexes 
belongs to each cluster according to distance matrix `Δ`.
+ `niters` is the number of total iterations
+ if `verbose`, print the k-medoids loss
"""
function kcentres(Δ::Matrix{T}, K::Int; niters::Int=5, verbose::Bool=false) where T
    c₁, L₁ = kcentreids(Δ, K; niters, verbose)
    c₂, L₂ = kcentreids(Δ, K; niters, verbose)
    c = L₁ < L₂ ? c₁ : c₂
    idxmin = argmin(Δ[c,:], dims=1)
    cids = vec(@. first(Tuple(idxmin)))
    kids = Vector{Vector{Int}}(undef,K)
    for k ∈ 1:K
        # idxs belong to cluster k, i.e. [1,3,9]
        kids[k] = findall(u->u==k, cids)
    end
    return c, kids
end


"""
    kclusters(Δ::Matrix{T},
              K::Int;
         niters::Int=5,
        verbose::Bool=false) -> vecids_of_K_centers::Vector{Vector{Int}}
Return samples indexes belongs to each cluster according to distance matrix `Δ`.
+ `niters` is the number of total iterations
+ if `verbose`, print the k-medoids loss
"""
function kclusters(Δ::Matrix{T}, K::Int; niters::Int=5, verbose::Bool=false) where T
    c₁, L₁ = kcentreids(Δ, K; niters, verbose)
    c₂, L₂ = kcentreids(Δ, K; niters, verbose)
    c = L₁ < L₂ ? c₁ : c₂
    idxmin = argmin(Δ[c,:], dims=1)
    cids = vec(@. first(Tuple(idxmin)))
    kids = Vector{Vector{Int}}(undef,K)
    for k ∈ 1:K
        # idxs belong to cluster k, i.e. [1,3,9]
        kids[k] = findall(u->u==k, cids)
    end
    return kids
end



