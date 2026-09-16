"""
    Distances that build mercer kernel
"""
const Mercer = Union{CosineDist,CorrDist}


"""
    Conditionally Negative Definite Distances
"""
const CondNegDefDist = Union{
    Euclidean,WeightedEuclidean,
    SqEuclidean,WeightedSqEuclidean,
    Cityblock,WeightedCityblock,
    Chebyshev,
    Minkowski,WeightedMinkowski, # 1≤p≤2
    ChiSqDist,
    Mahalanobis,SqMahalanobis,
    BhattacharyyaDist,
    HellingerDist,
    Haversine,
    SphericalAngle}


"""
    Check if the Distances type is Positive Semi-Definite
"""
function ispsd(::Type{Mercer})
    return true
end

function ispsd(::Type{CondNegDefDist})
    return false
end



