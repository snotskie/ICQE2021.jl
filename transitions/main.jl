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

Plots.scalefontsizes()
Plots.scalefontsizes(2)

let # create a non-global scope

# Data
data = DataFrame(CSV.File("data/data.csv", normalizenames=true))
data[!, :LABEL] = repeat(["No Label"], nrow(data))
derivedCode!(data, :BODY, :Changes, :Mood, :Oily, :Dysphoria, :Cry)
derivedCode!(data, :REFLECT, :Identity, :Longing, :Dream, :Childhood, :Family, :Name, :Letter, :Doubt, :Religion)
derivedCode!(data, :LEARN, :WWW, :Experiment, :Recipe)
derivedCode!(data, :PROGRESS, :Strangers, :Passed, :Out, :Affirmation)

# Config
codes = [
    :BODY,
    # :Changes, # derived into :BODY
    # :Mood, # derived into :BODY
    # :Dysphoria, # derived into :BODY
    # :Oily, # derived into :BODY
    # :Cry, # derived into :BODY
    :REFLECT,
    # :Identity, # derived into :REFLECT
    # :Longing, # derived into :REFLECT
    # :Dream, # derived into :REFLECT
    # :Childhood, # derived into :REFLECT
    # :Family, # derived into :REFLECT
    # :Name, # derived into :REFLECT
    # :Letter, # derived into :REFLECT
    # :Doubt, # derived into :REFLECT
    # :Religion # derived into :REFLECT
    :LEARN,
    # :WWW, # derived into :LEARN
    # :Experiment, # derived into :LEARN
    # :Recipe, # derived into :LEARN
    :PROGRESS,
    # :Strangers, # derived into :PROGRESS
    # :Out, # derived into :PROGRESS
    # :Passed, # derived into :PROGRESS
    # :Affirmation, # derived into :PROGRESS
    :DoseTracking,
    :SkippedDose,
    :Happy,
    :NonHappy,
    :Sweets,
    # :Friends, # qualitatively does not tell us anything
]
conversations = [:Day]
units = [:Day]
dropEmpty=true
sphereNormalize=true
seed = 4321
knn = 35
min_cluster_size=10
min_neighbors=2
limses = [0.025, 0.05, 0.1]
colorMap = Dict("No Label" => colorant"black")

# ENA
enaSVD = ENAModel(data, codes, conversations, units, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
p = plot(enaSVD, weakLinks=false, lims=0.75)
savefig(p, "images/SVD.png")
display(enaSVD)
display(p)

rotation = FormulaRotation(LinearModel, 2, @formula(col ~ 1 + Day), nothing)
ena = ENAModel(data, codes, conversations, units, rotateBy=rotation, dropEmpty=dropEmpty, sphereNormalize=sphereNormalize)
p = plot(ena, weakLinks=false, lims=0.75, xlabel="Day", ylabel="SVD1'")
savefig(p, "images/F1.png")             
display(p)

agg_data = combine(groupby(ena.metadata, :LABEL), sort(codes) .=> sum .=> sort(codes))
CSV.write("data/agg_data.csv", agg_data)
display(agg_data)

# The Gamut
function gamut(epsval, w, xlims, ylims)

    ## UMAP
    model = embedUnits!(ena, :Day, knn, w, seed)
    embedNetwork!(ena, model, seed)

    ## DBSCAN
    autocluster!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors)

    ## Plotting
    p = plot(ena, weakLinks=false, groupBy=:LABEL, lims=0.75, xlabel="Day", ylabel="SVD1'")
    savefig(p, "images/LabelF1_$(w).png")
    display(p)

    p = plotUMAP(ena, colorMap, :Day, xlims=xlims, ylims=ylims)
    savefig(p, "images/SpectralUMAP_$(w).png")
    display(p)

    p = plotUMAP(ena, colorMap, :Day, colormode=:label, xlims=xlims, ylims=ylims)
    savefig(p, "images/LabelUMAP_$(w).png")
    display(p)

    for (i, group) in enumerate(sort(unique(ena.metadata[!, :LABEL])))
        if group != "No Label"
            p = plotUMAP(ena, colorMap, :Day, colormode=:label, group=group, xlims=xlims, ylims=ylims)
            savefig(p, "images/GroupedUMAP_$(w)_group_$(i).png")
            display(p)
        end
    end

    ## Descriptive Statistics
    bounds = findLabelBounds(ena, :Date)
    CSV.write("data/bounds_$(w).csv", bounds)
    display(bounds)

    agg_data = combine(groupby(ena.metadata, :LABEL), sort(codes) .=> sum .=> sort(codes))
    CSV.write("data/agg_data_$(w).csv", agg_data)
    display(agg_data)

    p = plotCDFs(ena, :Day, colorMap)
    savefig(p, "images/CDFs_$(w).png")
    display(p)
end

gamut(0.6, 0.0, (-1.5, 1), (-1, 1.5))
gamut(0.5, 0.999999999999, (-1.6, 1), (-1, 1.6))
gamut(0.375, 1 / (nrow(ena.networkModel) + 1), (-1, 1.5), (-1, 1.5))

# A specific subsetted F1 rotation building off that last gamut
ena = ENAModel(
    data, codes, conversations, units,
    rotateBy=rotation,
    dropEmpty=dropEmpty,
    sphereNormalize=sphereNormalize,
    subsetFilter=(row->row[:LABEL] in [
        # "Auto Cluster #1",
        # "Auto Cluster #2",
        # "Auto Cluster #5",
        # "Auto Cluster #6"
        "#1",
        "#2",
        "#5",
        "#6"
    ])
)

p = plot(ena, weakLinks=false, groupBy=:LABEL, extraColors=EpistemicNetworkAnalysis.DEFAULT_EXTRA_COLORS[[1, 2, 5, 6]], lims=0.75, xlabel="Day", ylabel="SVD1'")
savefig(p, "images/SubsetF1.png")
display(p)

cluster1rows = [row[:LABEL] == "#1" for row in eachrow(ena.metadata)]
cluster2rows = [row[:LABEL] == "#2" for row in eachrow(ena.metadata)]
cluster5rows = [row[:LABEL] == "#5" for row in eachrow(ena.metadata)]
cluster6rows = [row[:LABEL] == "#6" for row in eachrow(ena.metadata)]
# cluster1rows = [row[:LABEL] == "Auto Cluster #1" for row in eachrow(ena.metadata)]
# cluster2rows = [row[:LABEL] == "Auto Cluster #2" for row in eachrow(ena.metadata)]
# cluster5rows = [row[:LABEL] == "Auto Cluster #5" for row in eachrow(ena.metadata)]
# cluster6rows = [row[:LABEL] == "Auto Cluster #6" for row in eachrow(ena.metadata)]
result12 = MannWhitneyUTest(ena.accumModel[cluster1rows, :pos_y], ena.accumModel[cluster2rows, :pos_y])
result15 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster1rows, :pos_x])
result56 = MannWhitneyUTest(ena.accumModel[cluster5rows, :pos_x], ena.accumModel[cluster6rows, :pos_x])
display(result12)
display(result15)
display(result56)
display(median(ena.accumModel[cluster1rows, :pos_y]))
display(median(ena.accumModel[cluster2rows, :pos_y]))
display(median(ena.accumModel[cluster1rows, :pos_x]))
display(median(ena.accumModel[cluster5rows, :pos_x]))
display(median(ena.accumModel[cluster6rows, :pos_x]))
display(ena)



# # LDA
# groups = sort(unique(data[!, :LABEL]))
# for dim1 in 1:(length(groups)-3)
#     ## Run and plot LDA for all nodes
#     rotation = LDARotation(:LABEL, dim1)
#     ena = ENAModel(data, codes, conversations, units, rotateBy=rotation,
#         dropEmpty=dropEmpty, sphereNormalize=sphereNormalize,
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
#         dropEmpty=dropEmpty, sphereNormalize=sphereNormalize,
#         subsetFilter=x->x[:LABEL]!="No Label") # TODO fix this

#     p = plot(ena, weakLinks=false)
#     savefig(p, "images/MR_$(group1)_$(group2).png")
#     # TODO run mann whitney tests, and pull out and report the coregistrations
# end



end # let