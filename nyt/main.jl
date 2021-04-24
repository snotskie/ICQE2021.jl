using EpistemicNetworkAnalysis
using Plots
using LinearAlgebra
using DataFrames
using Statistics
using CSV
using GLM
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

# Biplot
## Run model
ena = BiplotModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
display(ena)

## Plot
p = plot(ena)
savefig(p, "images/Biplot.png")

# SVD
## Run model
ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
display(ena)

## Plot normally
p = plot(ena)
savefig(p, "images/SVD.png")

## Plot with warps shown
p = plot(ena, showWarps=true)
savefig(p, "images/SVD_Warped.png")

## Plot hiding weak links
p = plot(ena, weakLinks=false)
savefig(p, "images/SVD_NoWeakLinks.png")

# Means Rotation, comparing lower word counts to higher word counts
rotation = MeansRotation(:WCGroup, "Lower Half", "Upper Half")
ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
display(ena)
p = plot(ena)
savefig(p, "images/MR1a.png")
p = plot(ena, showWarps=true)
savefig(p, "images/MR1a_Warped.png")
p = plot(ena, weakLinks=false)
savefig(p, "images/MR1a_NoWeakLinks.png")

# Means Rotation, those that showed negative emotions vs. those that didn't
rotation = MeansRotation(:NegString, "0", "1")
ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
display(ena)
p = plot(ena)
savefig(p, "images/MR1b.png")
p = plot(ena, showWarps=true)
savefig(p, "images/MR1b_Warped.png")
p = plot(ena, weakLinks=false)
savefig(p, "images/MR1b_NoWeakLinks.png")

# Means Rotation, those that showed positive emotions vs. those that didn't
rotation = MeansRotation(:PosString, "0", "1")
ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
display(ena)
p = plot(ena)
savefig(p, "images/MR1c.png")
p = plot(ena, showWarps=true)
savefig(p, "images/MR1c_Warped.png")
p = plot(ena, weakLinks=false)
savefig(p, "images/MR1c_NoWeakLinks.png")

# cluster1rows = [row[:LABEL] == "Auto Cluster #1" for row in eachrow(ena.metadata)]
# cluster2rows = [row[:LABEL] == "Auto Cluster #2" for row in eachrow(ena.metadata)]
# cluster5rows = [row[:LABEL] == "Auto Cluster #5" for row in eachrow(ena.metadata)]
# cluster6rows = [row[:LABEL] == "Auto Cluster #6" for row in eachrow(ena.metadata)]
# result12 = MannWhitneyUTest(ena.accumModel[cluster1rows, :pos_y], ena.accumModel[cluster2rows, :pos_y])
# result15 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster1rows, :pos_x])
# result56 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster6rows, :pos_x])

end # let