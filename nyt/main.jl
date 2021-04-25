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


# derivedCode!(data, 2, :BITTER, :Policy, :Normalcy)
# derivedCode!(data, 2, :SOFT, :Management, :Normalcy)
# derivedCode!(data, 2, :LTS, :Learning, :Technology, :SharedSpace)
# mc_median = median(data[!, :WordCount])
# data[!, :WCGroup] = map(data[!, :WordCount]) do wc
#     if wc < mc_median
#         return "Lower Half"
#     else
#         return "Upper Half"
#     end
# end

# data[!, :EMOTION] = map(eachrow(data)) do row
#     if row[:PositiveVE] > 0 && row[:NegativeVE] == 0
#         return "+ve"
#     elseif row[:PositiveVE] == 0 && row[:NegativeVE] > 0
#         return "-ve"
#     elseif row[:PositiveVE] > 0 && row[:NegativeVE] > 0
#         return "±ve"
#     else
#         return "0ve"
#     end
# end


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



# Auto-Clustering
data[!, :LABEL] = repeat(["No Label"], nrow(data))
colorMap = Dict("No Label" => colorant"black")
ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
function autocluster!(ena, data, colorMap, k)

    ### Prepare data
    X = Matrix{Float64}(ena.accumModel[!, ena.networkModel[!, :relationship]])
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
end

for k in 2:10
    autocluster!(ena, data, colorMap, k)
    p = plot(ena, groupBy=:LABEL, showNetworks=true, weakLinks=false)
    savefig(p, "images/SVD-$(k).png")
    p = plot(ena, groupBy=:LABEL, showNetworks=false)
    savefig(p, "images/SVD-$(k)-NoNetwork.png")

    rotation = LDARotation(:LABEL)
    enaLDA = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
    p = plot(enaLDA, groupBy=:LABEL, showNetworks=true, lims=0.5)
    savefig(p, "images/LDA-$(k).png")
    p = plot(enaLDA, groupBy=:LABEL, showNetworks=false, lims=0.5)
    savefig(p, "images/LDA-$(k)-NoNetworks.png")
end



# # Biplot
# ## Run model
# ena = BiplotModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
# display(ena)

# ## Plot
# p = plot(ena)
# savefig(p, "images/Biplot.png")

# # SVD
# ## Run model
# ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
# display(ena)

# ## Plot normally
# p = plot(ena)
# savefig(p, "images/SVD.png")

# ## Plot with warps shown
# p = plot(ena, showWarps=true)
# savefig(p, "images/SVD_Warped.png")

# ## Plot hiding weak links
# p = plot(ena, weakLinks=false)
# savefig(p, "images/SVD_NoWeakLinks.png")

# # Means Rotation, comparing lower word counts to higher word counts
# rotation = MeansRotation(:WCGroup, "Lower Half", "Upper Half")
# ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
# display(ena)
# p = plot(ena)
# savefig(p, "images/MR1a.png")
# p = plot(ena, showWarps=true)
# savefig(p, "images/MR1a_Warped.png")
# p = plot(ena, weakLinks=false)
# savefig(p, "images/MR1a_NoWeakLinks.png")

# # Means Rotation, those that showed negative emotions vs. those that didn't
# rotation = MeansRotation(:NegativeVE, 0, 1)
# ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
# display(ena)
# p = plot(ena)
# savefig(p, "images/MR1b.png")
# p = plot(ena, showWarps=true)
# savefig(p, "images/MR1b_Warped.png")
# p = plot(ena, weakLinks=false)
# savefig(p, "images/MR1b_NoWeakLinks.png")

# # Means Rotation, those that showed positive emotions vs. those that didn't
# rotation = MeansRotation(:PositiveVE, 0, 1)
# ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
# display(ena)
# p = plot(ena)
# savefig(p, "images/MR1c.png")
# p = plot(ena, showWarps=true)
# savefig(p, "images/MR1c_Warped.png")
# p = plot(ena, weakLinks=false)
# savefig(p, "images/MR1c_NoWeakLinks.png")

# # Means Rotation, testing an idea
# rotation = MeansRotation(:LTS, 0, 1)
# ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation,
#     relationshipFilter=(i,j,ci,cj)->(i<j&&!((i,j) in [(5,8), (5,10), (8,10)])))
# display(ena)
# p = plot(ena)
# savefig(p, "images/MR1d.png")
# p = plot(ena, showWarps=true)
# savefig(p, "images/MR1d_Warped.png")
# p = plot(ena, weakLinks=false)
# savefig(p, "images/MR1d_NoWeakLinks.png")

# # Means Rotation, testing an idea
# rotation = MeansRotation(:BITTER, 0, 1)
# ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation,
#     relationshipFilter=(i,j,ci,cj)->(i<j&&(i,j) != (3,11)))
# display(ena)
# p = plot(ena)
# savefig(p, "images/MR1e.png")
# p = plot(ena, showWarps=true)
# savefig(p, "images/MR1e_Warped.png")
# p = plot(ena, weakLinks=false)
# savefig(p, "images/MR1e_NoWeakLinks.png")

# # Means Rotation, testing an idea
# rotation = MeansRotation(:SOFT, 0, 1)
# ena = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation,
#     relationshipFilter=(i,j,ci,cj)->(i<j&&(i,j) != (3,12)))
# display(ena)
# p = plot(ena)
# savefig(p, "images/MR1f.png")
# p = plot(ena, showWarps=true)
# savefig(p, "images/MR1f_Warped.png")
# p = plot(ena, weakLinks=false)
# savefig(p, "images/MR1f_NoWeakLinks.png")

# # Means Rotation, testing something
# # rotation = MeansRotation(:EMOTION, "-ve", "+ve")
# rotation = Means2Rotation(:PositiveVE, 0, 1, :NegativeVE, 0, 1)
# ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
# display(ena)
# p = plot(ena)
# savefig(p, "images/MR1g.png")
# p = plot(ena, showWarps=true)
# savefig(p, "images/MR1g_Warped.png")
# p = plot(ena, weakLinks=false)
# savefig(p, "images/MR1g_NoWeakLinks.png")

# # Means Rotation, testing something
# # rotation = MeansRotation(:EMOTION, "-ve", "+ve")
# rotation = Means2Rotation(:NegativeVE, 0, 1, :PositiveVE, 0, 1)
# ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, rotateBy=rotation)
# display(ena)
# p = plot(ena)
# savefig(p, "images/MR1h.png")
# p = plot(ena, showWarps=true)
# savefig(p, "images/MR1h_Warped.png")
# p = plot(ena, weakLinks=false)
# savefig(p, "images/MR1h_NoWeakLinks.png")




# # The Gamut
# seed = 4321
# knn = 35
# min_cluster_size=10
# min_neighbors=2
# colorMap = Dict("No Label" => colorant"black")
# function gamut(epsval, w)

#     ## ENA
#     ena = ENAModel(data, codes[3:end], conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)

#     ## UMAP
#     model = embedUnits!(ena, :WordCount, knn, w, seed)
#     embedNetwork!(ena, model, seed)

#     ## DBSCAN
#     autocluster!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors)

#     ## Plotting
#     p = plot(ena, weakLinks=false, groupBy=:LABEL)
#     savefig(p, "images/LabelF1_$(w).png")
#     display(p)

#     p = plotUMAP(ena, colorMap, :WordCount)
#     savefig(p, "images/SpectralUMAP_$(w).png")
#     display(p)

#     p = plotUMAP(ena, colorMap, :WordCount, colormode=:label)
#     savefig(p, "images/LabelUMAP_$(w).png")
#     display(p)

#     for (i, group) in enumerate(sort(unique(ena.metadata[!, :LABEL])))
#         if group != "No Label"
#             p = plotUMAP(ena, colorMap, :WordCount, colormode=:label, group=group)
#             savefig(p, "images/GroupedUMAP_$(w)_group_$(i).png")
#             display(p)
#         end
#     end

#     # ## Descriptive Statistics
#     # bounds = findLabelBounds(ena, :Date)
#     # CSV.write("data/bounds_$(w).csv", bounds)
#     # display(bounds)

#     # agg_data = combine(groupby(ena.metadata, :LABEL), sort(codes) .=> sum .=> sort(codes))
#     # CSV.write("data/agg_data_$(w).csv", agg_data)
#     # display(agg_data)

#     # p = plotCDFs(ena, :Day, colorMap)
#     # savefig(p, "images/CDFs_$(w).png")
#     # display(p)
# end

# gamut(0.45, 0.0)
# gamut(0.375, 1 / (nrow(ena.networkModel) + 1))
# gamut(0.375, 0.999999)


# cluster1rows = [row[:LABEL] == "Auto Cluster #1" for row in eachrow(ena.metadata)]
# cluster2rows = [row[:LABEL] == "Auto Cluster #2" for row in eachrow(ena.metadata)]
# cluster5rows = [row[:LABEL] == "Auto Cluster #5" for row in eachrow(ena.metadata)]
# cluster6rows = [row[:LABEL] == "Auto Cluster #6" for row in eachrow(ena.metadata)]
# result12 = MannWhitneyUTest(ena.accumModel[cluster1rows, :pos_y], ena.accumModel[cluster2rows, :pos_y])
# result15 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster1rows, :pos_x])
# result56 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster6rows, :pos_x])

end # let