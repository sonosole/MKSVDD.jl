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

include("preimage.jl")
export rbfpreimage, trackx2y

include("centre.jl")
export centre, rbfcentre

include("cluster.jl")
export distmat, kcentreids, kcentres, kclusters

include("approx.jl")
export rbfksvdd

include("train-smo.jl")
export smosvdd

include("train-cosmo.jl")
export svdd, svddlabel


# end # module MKSVDD
