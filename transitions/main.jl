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

# Prepare data
data = DataFrame(CSV.File("data/data.csv", normalizenames=true))
data[!, :LABEL] = repeat(["No Label"], nrow(data))

# Model config
codes = [
    :WWW,
    :Identity,
    :Longing,
    :Changes,
    # :Mood, # same as :Strangers
    # :Strangers, # least qualitatively insightful, curse of dimensionality
    :Dream,
    :Childhood,
    :Experiment,
    # :Dysphoria, # same as :Strangers
    :Affirmation,
    :Family,
    :Name,
    :Letter,
    :Recipe,
    :DoseTracking,
    :SkippedDose,
    :Happy,
    :NonHappy,
    :Sweets,
    :Oily,
    # :Friends, # qualitatively does not tell us anything
    :Out,
    :Doubt,
    :Cry,
    :Passed,
    # :Religion # only appears once in the data
]
conversations = [:Day]
units = [:Day]
dropEmpty=true
sphereNormalize=false
dimensionNormalize=true
seed = 4321
weight1 = 0.999999999999
knn = 35
epsval = 0.4
min_cluster_size=5
min_neighbors=2
limses = [0.025, 0.05, 0.1]
colorMap = Dict("No Label" => colorant"black")

# Descriptive Statistics
println("Descripive Statistics:")
for code in sort(codes)
    println("  $(code): $(sum(data[!, code]))")
end

# ENA
enaSVD = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, dimensionNormalize=dimensionNormalize)
p = plot(enaSVD, weakLinks=false)
savefig(p, "images/SVD.png")
display(p)

rotation = FormulaRotation(LinearModel, 2, @formula(col ~ 1 + Day), nothing)
ena = ENAModel(data, codes, conversations, units, rotateBy=rotation, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, dimensionNormalize=dimensionNormalize)
p = plot(ena, weakLinks=false)
savefig(p, "images/F1.png")
display(p)

# No weight1
## UMAP
model = embedUnits!(ena, :Day, knn, 0.0, seed)
embedNetwork!(ena, model, seed)

## DBSCAN
autocluster!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors)

## Plotting
p = plot(ena, weakLinks=false, groupBy=:LABEL)
savefig(p, "images/LabelF1_0.0.png")
display(p)
p = plotUMAP(ena, colorMap, :Day)
savefig(p, "images/SpectralUMAP_0.0.png")
display(p)
p = plotUMAP(ena, colorMap, :Day, colormode=:label)
savefig(p, "images/LabelUMAP_0.0.png")
display(p)
for (i, group) in enumerate(sort(unique(ena.metadata[!, :LABEL])))
    if group != "No Label"
        p = plotUMAP(ena, colorMap, :Day, colormode=:label, group=group)
        savefig(p, "images/GroupedUMAP_0.0_group_$(i).png")
        display(p)
    end
end

# Low weight1
w = 1 / (nrow(ena.networkModel) + 1)

## UMAP
model = embedUnits!(ena, :Day, knn, w, seed)
embedNetwork!(ena, model, seed)

## DBSCAN
autocluster!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors)

## Plotting
p = plot(ena, weakLinks=false, groupBy=:LABEL)
savefig(p, "images/LabelF1_$(w).png")
display(p)
p = plotUMAP(ena, colorMap, :Day)
savefig(p, "images/SpectralUMAP_$(w).png")
display(p)
p = plotUMAP(ena, colorMap, :Day, colormode=:label)
savefig(p, "images/LabelUMAP_$(w).png")
display(p)
for (i, group) in enumerate(sort(unique(ena.metadata[!, :LABEL])))
    if group != "No Label"
        p = plotUMAP(ena, colorMap, :Day, colormode=:label, group=group)
        savefig(p, "images/GroupedUMAP_$(w)_group_$(i).png")
        display(p)
    end
end

# Target weight1
## UMAP
model = embedUnits!(ena, :Day, knn, weight1, seed)
embedNetwork!(ena, model, seed)

## DBSCAN
autocluster!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors)

## Plotting
p = plot(ena, weakLinks=false, groupBy=:LABEL)
savefig(p, "images/LabelF1_$(weight1).png")
display(p)
p = plotUMAP(ena, colorMap, :Day)
savefig(p, "images/SpectralUMAP_$(weight1).png")
display(p)
p = plotUMAP(ena, colorMap, :Day, colormode=:label)
savefig(p, "images/LabelUMAP_$(weight1).png")
display(p)
for (i, group) in enumerate(sort(unique(ena.metadata[!, :LABEL])))
    if group != "No Label"
        p = plotUMAP(ena, colorMap, :Day, colormode=:label, group=group)
        savefig(p, "images/GroupedUMAP_$(weight1)_group_$(i).png")
        display(p)
    end
end



# ## Displaying time windows (text)
# for label in sort(unique(data[!, :LABEL]))
#     if label != "No Label"
#         labelRows = data[!, :LABEL] .== label
#         println("$(label): $(first(data[labelRows, :Date])) -- $(last(data[labelRows, :Date]))")
#     end
# end

# ## Displaying time windows (cdf's)
# dayLabelMap = Dict(row[:Day] => row[:LABEL] for row in eachrow(data))
# labelCounts = Dict(label => 0 for label in unique(data[!, :LABEL]))
# labelPrevX = Dict(label => 0 for label in unique(data[!, :LABEL]))
# p = plot(; size=(800,800))
# for (x, day) in enumerate(data[!, :Day])
#     label = dayLabelMap[day]
#     if label != "No Label"
#         plot!(p,
#             [labelPrevX[label], x],
#             [labelCounts[label], labelCounts[label] + 1],
#             label=nothing,
#             seriestype=:line,
#             linecolor=colorMap[label])

#         labelCounts[label] += 1
#         labelPrevX[label] = x
#     end
# end

# savefig(p, "images/CDFs.png")

# # Code and Count
# agg_data = combine(groupby(data, :LABEL), sort(codes) .=> sum .=> sort(codes))
# display(agg_data)
# CSV.write("data/agg_data.csv", agg_data)

# # LDA
# groups = sort(unique(data[!, :LABEL]))
# for dim1 in 1:(length(groups)-3)
#     ## Run and plot LDA for all nodes
#     rotation = LDARotation(:LABEL, dim1)
#     ena = ENAModel(data, codes, conversations, units, rotateBy=rotation,
#         dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, dimensionNormalize=dimensionNormalize,
#         subsetFilter=x->x[:LABEL]!="No Label")
        
#     for lims in limses
#         p = plot(ena, weakLinks=false, showUnits=false, lims=lims)    
#         savefig(p, "images/LDA$(dim1)-$(lims).png")
#     end
# end

# # MR
# groups = sort(unique(data[!, :LABEL]))
# for group1 in 1:(length(groups)-2)
#     group2 = group1 + 1
#     rotation = MeansRotation(:LABEL, "Auto Cluster #$(group1)", "Auto Cluster #$(group2)")
#     ena = ENAModel(data, codes, conversations, units, rotateBy=rotation,
#         dropEmpty=dropEmpty, sphereNormalize=sphereNormalize, dimensionNormalize=dimensionNormalize,
#         subsetFilter=x->x[:LABEL]!="No Label") # TODO fix this

#     p = plot(ena, weakLinks=false)
#     savefig(p, "images/MR_$(group1)_$(group2).png")
#     # TODO run mann whitney tests, and pull out and report the coregistrations
# end



end # let