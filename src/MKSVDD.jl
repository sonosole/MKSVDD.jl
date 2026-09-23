# module MKSVDD

using Distances
using JuMP
using COSMO
using Random

import LinearAlgebra: diag, lmul!, \

include("rules.jl")
export Mercer
export CondNegDefDist
export ispsd

include("SVDD.jl")
export SVDD
export radius, radius², nsvs, svs, alphasᵀ, alphas, cdotc, kernelf
export minratio, maxratio
export absratio, abs2ratio, sqdiff, svddprob

include("SVDD-preimage.jl")
export rbfpreimage

include("SVDD-centre.jl")
export centre, rbfcentre

include("SVDD-cluster.jl")
export distmat, kcentreids, kcentres, kclusters

include("SVDD-approx.jl")
export rbfksvdd

include("SVDD-SMO.jl")
export smosvdd

include("SVDD-Optimizer.jl")
export svdd, svddlabel


# end # module MKSVDD
