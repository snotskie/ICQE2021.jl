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

let # create a non-global scope

# Data
data = DataFrame(CSV.File("data/data.csv", normalizenames=true))
mc_median = median(data[!, :WordCount])
data[!, :WCGroup] = map(data[!, :WordCount]) do wc
    if wc < mc_median
        return "Lower Half"
    else
        return "Upper Half"
    end
end

data[!, :NegString] = map(string, data[!, :NegativeVE])
data[!, :PosString] = map(string, data[!, :PositiveVE])

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
weakLinks=true
groupBy=:WCGroup

# Biplot
ena = BiplotModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
p = plot(ena)
savefig(p, "images/Biplot.png")
display(ena)
display(p)

# SVD
ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
p = plot(ena, weakLinks=weakLinks)
savefig(p, "images/SVD.png")
display(ena)
display(p)

# Means Rotation, comparing lower word counts to higher word counts
rotation = MeansRotation(:WCGroup, "Lower Half", "Upper Half")
ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
p = plot(ena, weakLinks=weakLinks)
savefig(p, "images/MR1a.png")
display(ena)
display(p)

rotation = MeansRotation(:NegString, "0", "1")
ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
p = plot(ena, weakLinks=weakLinks)
savefig(p, "images/MR1b.png")
display(ena)
display(p)

rotation = MeansRotation(:PosString, "0", "1")
ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
p = plot(ena, weakLinks=weakLinks)
savefig(p, "images/MR1c.png")
display(ena)
display(p)

# cluster1rows = [row[:LABEL] == "Auto Cluster #1" for row in eachrow(ena.metadata)]
# cluster2rows = [row[:LABEL] == "Auto Cluster #2" for row in eachrow(ena.metadata)]
# cluster5rows = [row[:LABEL] == "Auto Cluster #5" for row in eachrow(ena.metadata)]
# cluster6rows = [row[:LABEL] == "Auto Cluster #6" for row in eachrow(ena.metadata)]
# result12 = MannWhitneyUTest(ena.accumModel[cluster1rows, :pos_y], ena.accumModel[cluster2rows, :pos_y])
# result15 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster1rows, :pos_x])
# result56 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster6rows, :pos_x])

end # let