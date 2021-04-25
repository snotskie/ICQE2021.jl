using EpistemicNetworkAnalysis
using UMAP
using Plots
using LinearAlgebra
using DataFrames
using Random
using Statistics
using CSV
using Distances
using Dates
using Colors
using GLM
using Clustering
using HypothesisTests

cd(Base.source_dir())
include("helpers.jl")

let # create a non-global scope

# Data
data = DataFrame(CSV.File("data/data.csv", normalizenames=true))

# Config
codes = [
    :NegativeVE,
    :PositiveVE,
    :Normalcy,
    :Safety,
    :SharedSpace,
    :ComingOfAge,
    :Friends,
    :Learning,
    :Mental,
    :Technology,
    :Policy,
    :Management
]

conversations = [:ThreadNumber]
units = [:ThreadNumber, :CommentPosition]
dropEmpty=true
sphereNormalize=true



# K-Means clustering approach
data[!, :LABEL] = repeat(["No Label"], nrow(data))
colorMap = Dict("No Label" => colorant"black")
ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
function autocluster!(ena, data, colorMap, k, dims)

    ### Prepare data
    X = Matrix{Float64}(ena.accumModel[!, dims])
    X = Matrix{Float64}(transpose(X))
    
    ### Using k-means to detect labels
    results = kmeans(X, k)

    ### Applying the labels
    ena.metadata[!, :LABEL] .= "No Label"
    for (i, cluster) in enumerate(results.assignments)
        mylabel = "Auto Cluster #$(cluster)"
        ena.metadata[i, :LABEL] = mylabel
        if !haskey(colorMap, mylabel)
            if i <= length(EpistemicNetworkAnalysis.DEFAULT_EXTRA_COLORS)
                colorMap[mylabel] = EpistemicNetworkAnalysis.DEFAULT_EXTRA_COLORS[i]
            else
                colorMap[mylabel] = RGB(rand(0.4:0.01:0.9), rand(0.4:0.01:0.9), rand(0.4:0.01:0.9))
            end
        end
    end

    ### Saving labels back into original dataframe
    data[!, :LABEL] .= "No Label"
    for row in eachrow(ena.metadata)
        dataRows = [join(dataRow[ena.units], ".") == row[:ENA_UNIT] for dataRow in eachrow(data)]
        data[dataRows, :LABEL] = row[:LABEL]
    end

    ### Return score
    SSE = 0
    for (i, cluster) in enumerate(results.assignments)
        SSE += evaluate(Euclidean(), X[:, i], results.centers[:, cluster]) ^ 2
    end

    return SSE
end

K = 5
for k in 1:K
    autocluster!(ena, data, colorMap, k, ena.networkModel[!, :relationship])
    p = plot(ena, groupBy=:LABEL, showNetworks=true, weakLinks=false, leg=false)
    savefig(p, "images/SVD-$(k).png")
    p = plot(ena, groupBy=:LABEL, showNetworks=false, leg=false)
    savefig(p, "images/SVD-$(k)-NoNetwork.png")
end



# # DBSCAN approach
# epsval = 0.07
# min_cluster_size=10
# min_neighbors=2
# autocluster2!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors, [:pos_x, :pos_y])
# p = plot(ena, groupBy=:LABEL, showNetworks=true, weakLinks=false, leg=false)
# savefig(p, "images/SVD-DBSCAN.png")
# p = plot(ena, groupBy=:LABEL, showNetworks=false, leg=false)
# savefig(p, "images/SVD-DBSCAN-NoNetwork.png")



# Quandrant approach
ena.metadata[!, :LABEL] = map(eachrow(ena.accumModel)) do row
    if row[:pos_y] > 0
        if row[:pos_x] > 0
            return "I"
        else
            return "II"
        end
    else
        if row[:pos_x] > 0
            return "IV"
        else
            return "III"
        end
    end
end

data[!, :LABEL] .= "No Label"
for row in eachrow(ena.metadata)
    dataRows = [join(dataRow[ena.units], ".") == row[:ENA_UNIT] for dataRow in eachrow(data)]
    data[dataRows, :LABEL] = row[:LABEL]
end

p = plot(ena, groupBy=:LABEL, showNetworks=true, weakLinks=false, leg=false)
savefig(p, "images/SVD-Quadrant.png")
p = plot(ena, groupBy=:LABEL, showNetworks=false, leg=false)
savefig(p, "images/SVD-Quadrant-NoNetwork.png")

# rotation = LDARotation(:LABEL)
# ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
# p = plot(ena, groupBy=:LABEL, showNetworks=true, weakLinks=false, leg=false, lims=0.25)
# savefig(p, "images/LDA-Quadrant.png")
# p = plot(ena, groupBy=:LABEL, showNetworks=false, leg=false, lims=0.25)
# savefig(p, "images/LDA-Quadrant-NoNetwork.png")



# # The Nonlinear Gamut
# seed = 4321
# knn = 35
# min_cluster_size=10
# min_neighbors=2
# colorMap = Dict("No Label" => colorant"black")
# function gamut(epsval, w)

#     ## ENA
#     ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)

#     ## UMAP
#     model = embedUnits!(ena, :WordCount, knn, w, seed)
#     embedNetwork!(ena, model, seed)

#     ## DBSCAN
#     autocluster2!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors)

#     ## Plotting
#     p = plot(ena, weakLinks=false, groupBy=:LABEL)
#     savefig(p, "images/SVD_UMAP.png")

#     p = plotUMAP(ena, colorMap, :WordCount, colormode=:label)
#     savefig(p, "images/LabelUMAP.png")

#     for (i, group) in enumerate(sort(unique(ena.metadata[!, :LABEL])))
#         if group != "No Label"
#             p = plotUMAP(ena, colorMap, :WordCount, colormode=:label, group=group)
#             savefig(p, "images/GroupedUMAP_$(i).png")
#         end
#     end
# end

# gamut(0.33, 1 / (nrow(ena.networkModel) + 1))


# cluster1rows = [row[:LABEL] == "Auto Cluster #1" for row in eachrow(ena.metadata)]
# cluster2rows = [row[:LABEL] == "Auto Cluster #2" for row in eachrow(ena.metadata)]
# cluster5rows = [row[:LABEL] == "Auto Cluster #5" for row in eachrow(ena.metadata)]
# cluster6rows = [row[:LABEL] == "Auto Cluster #6" for row in eachrow(ena.metadata)]
# result12 = MannWhitneyUTest(ena.accumModel[cluster1rows, :pos_y], ena.accumModel[cluster2rows, :pos_y])
# result15 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster1rows, :pos_x])
# result56 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster6rows, :pos_x])

println("Done!")

end # let