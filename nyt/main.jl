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
data[!, :LABEL] = repeat(["No Label"], nrow(data))
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
seed = 4321
knn = 35
min_cluster_size=10
min_neighbors=2
colorMap = Dict("No Label" => colorant"black")

# Biplot
## Run model
ena = BiplotModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
display(ena)

## Plot
p = plot(ena)
savefig(p, "images/Biplot.png")

# SVD
## Run model
ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
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
ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
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




# The Gamut
function gamut(epsval, w)

    ## ENA
    ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)

    ## UMAP
    model = embedUnits!(ena, :WordCount, knn, w, seed)
    embedNetwork!(ena, model, seed)

    ## DBSCAN
    autocluster!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors)

    ## Plotting
    p = plot(ena, weakLinks=false, groupBy=:LABEL)
    savefig(p, "images/LabelF1_$(w).png")
    display(p)

    p = plotUMAP(ena, colorMap, :WordCount)
    savefig(p, "images/SpectralUMAP_$(w).png")
    display(p)

    p = plotUMAP(ena, colorMap, :WordCount, colormode=:label)
    savefig(p, "images/LabelUMAP_$(w).png")
    display(p)

    for (i, group) in enumerate(sort(unique(ena.metadata[!, :LABEL])))
        if group != "No Label"
            p = plotUMAP(ena, colorMap, :WordCount, colormode=:label, group=group)
            savefig(p, "images/GroupedUMAP_$(w)_group_$(i).png")
            display(p)
        end
    end

    # ## Descriptive Statistics
    # bounds = findLabelBounds(ena, :Date)
    # CSV.write("data/bounds_$(w).csv", bounds)
    # display(bounds)

    # agg_data = combine(groupby(ena.metadata, :LABEL), sort(codes) .=> sum .=> sort(codes))
    # CSV.write("data/agg_data_$(w).csv", agg_data)
    # display(agg_data)

    # p = plotCDFs(ena, :Day, colorMap)
    # savefig(p, "images/CDFs_$(w).png")
    # display(p)
end

gamut(0.45, 0.0)
gamut(0.375, 1 / (nrow(ena.networkModel) + 1))
gamut(0.375, 0.999999)


# cluster1rows = [row[:LABEL] == "Auto Cluster #1" for row in eachrow(ena.metadata)]
# cluster2rows = [row[:LABEL] == "Auto Cluster #2" for row in eachrow(ena.metadata)]
# cluster5rows = [row[:LABEL] == "Auto Cluster #5" for row in eachrow(ena.metadata)]
# cluster6rows = [row[:LABEL] == "Auto Cluster #6" for row in eachrow(ena.metadata)]
# result12 = MannWhitneyUTest(ena.accumModel[cluster1rows, :pos_y], ena.accumModel[cluster2rows, :pos_y])
# result15 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster1rows, :pos_x])
# result56 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster6rows, :pos_x])

end # let