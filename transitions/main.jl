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

let # create a non-global scope

# Prepare data
data = DataFrame(CSV.File("data/data.csv", normalizenames=true))
data[!, :LABEL] = repeat(["No Label"], nrow(data))
data[!, :DayScaled] = (data[!, :Day] .- minimum(data[!, :Day])) / (maximum(data[!, :Day]) - minimum(data[!, :Day]))

# Model config
codes = [
    :WWW,
    :Identity,
    :Longing,
    :Changes,
    :Mood,
    :Strangers,
    :Dream,
    :Childhood,
    :Experiment,
    :Dysphoria,
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
    # # :Friends, # qualitatively does not tell us anything
    :Out,
    :Doubt,
    :Cry,
    :Passed,
    # # :Religion # only appears once in the data
]
conversations = [:Day]
units = [:Day]
seed = 4321
knn = 35
weights1 = [0.0, 0.01, 0.5, 0.7]
plotsizes = [7, 7, 20, 20]
epsval = 0.5
min_cluster_size=5
min_neighbors=2
limses = [0.025, 0.05, 0.1]
MRs = [(2, 6), (1, 5), (1, 3), (3, 4), (2, 4)]

# Descriptive Statistics
println("Descripive Statistics:")
for code in sort(codes)
    println("  $(code): $(sum(data[!, code]))")
end

# ENA
ena = ENAModel(data, codes, conversations, units, dropEmpty=true)
p = plot(ena, weakLinks=false)
savefig(p, "images/SVD.png")

# UMAP
## Preprocessing unit, code, and "line" data for embedding
centroidX = transpose(Matrix{Float64}(hcat(ena.metadata[!, :DayScaled]/maximum(ena.metadata[!, :DayScaled]), ena.centroidModel[!, ena.networkModel[!, :relationship]])))
codeX = transpose(Matrix{Float64}(hcat([-9999 for row in eachrow(ena.codeModel)], ena.codeModel[!, ena.networkModel[!, :relationship]])))
lineX = nothing
unitIndex = 0
codeIndex = size(centroidX, 2)
lineIndex = codeIndex + size(codeX, 2)
for (i, networkRow) in enumerate(eachrow(ena.networkModel))
    j, k = ena.relationshipMap[networkRow[:relationship]]
    pointA = Vector{Float64}([-9999, ena.codeModel[j, ena.networkModel[!, :relationship]]...])
    pointB = Vector{Float64}([-9999, ena.codeModel[k, ena.networkModel[!, :relationship]]...])
    point = 0.5*pointA + 0.5*pointB
    if isnothing(lineX)
        lineX = point
    else
        lineX = hcat(lineX, point)
    end
end

## Helper
function computeEmbedding(knn, weight1, seed)
    ### Seed
    Random.seed!(seed)

    ### Model the units
    weightn = (1-weight1) / nrow(ena.networkModel)
    weights = [weight1, [weightn for row in eachrow(ena.networkModel)]...]
    metric = WeightedEuclidean(weights/sum(weights))
    model = UMAP_(centroidX; n_neighbors=knn, min_dist=0.0000000001, metric=metric)

    ### Project network components into the embedding
    weightn = 1 / nrow(ena.networkModel)
    weights = [0, [weightn for row in eachrow(ena.networkModel)]...]
    metric = WeightedEuclidean(weights/sum(weights))
    networkEmbedding = UMAP.transform(model, hcat(codeX, lineX); n_neighbors=1, min_dist=0.0000000001, metric=metric)

    ### Unseed
    Random.seed!(Dates.value(now()))

    ### Concat and return
    embedding = hcat(model.embedding, networkEmbedding)
    return embedding
end

## Preprocessing colors and line widths for plotting
colorMap = Dict("No Label" => colorant"black")
allLineWidths = map(eachrow(ena.networkModel)) do networkRow
    return sum(ena.accumModel[!, networkRow[:relationship]])
end

## Helper
function plotUMAP(embedding, plotsize, colormode)
    p = plot(; size=(800,800))
    xticks!(p, [-plotsize, plotsize])
    yticks!(p, [-plotsize, plotsize])
    xlims!(p, (-plotsize, plotsize))
    ylims!(p, (-plotsize, plotsize))
    meanX = mean(embedding[1, 1:codeIndex])
    meanY = mean(embedding[2, 1:codeIndex])

    displayRows = repeat([true], nrow(ena.centroidModel))
    displayAccums = ena.accumModel[displayRows, :]
    lineWidths = map(eachrow(ena.networkModel)) do networkRow
        return sum(displayAccums[!, networkRow[:relationship]])
    end

    codeWidths = zeros(nrow(ena.codeModel))
    for (i, networkRow) in enumerate(eachrow(ena.networkModel))
        j, k = ena.relationshipMap[networkRow[:relationship]]
        codeWidths[j] += lineWidths[i]
        codeWidths[k] += lineWidths[i]
    end
    
    codeXs = zeros(nrow(ena.codeModel))
    codeYs = zeros(nrow(ena.codeModel))
    for (i, networkRow) in enumerate(eachrow(ena.networkModel))
        j, k = ena.relationshipMap[networkRow[:relationship]]
        # pointA = [embedding[1, codeIndex+j], embedding[2, codeIndex+j]]
        # pointB = [embedding[1, codeIndex+k], embedding[2, codeIndex+k]]
        pointT = [embedding[1, lineIndex+i], embedding[2, lineIndex+i]]
        codeXs[j] += pointT[1] * lineWidths[i] / codeWidths[j]
        codeXs[k] += pointT[1] * lineWidths[i] / codeWidths[k]
        codeYs[j] += pointT[2] * lineWidths[i] / codeWidths[j]
        codeYs[k] += pointT[2] * lineWidths[i] / codeWidths[k]
    end

    lineWidths *= 8 / maximum(allLineWidths)
    for (i, networkRow) in enumerate(eachrow(ena.networkModel))
        j, k = ena.relationshipMap[networkRow[:relationship]]
        # pointA = [embedding[1, codeIndex+j], embedding[2, codeIndex+j]]
        # pointB = [embedding[1, codeIndex+k], embedding[2, codeIndex+k]]
        pointA = [codeXs[j], codeYs[j]]
        pointB = [codeXs[k], codeYs[k]]
        pointT = [embedding[1, lineIndex+i], embedding[2, lineIndex+i]]
        points = hcat(pointA, pointT, pointT, pointT, pointB)
        plot!(p,
            points[1, :] .- meanX,
            points[2, :] .- meanY,
            label=nothing,
            seriestype=:curves,
            linewidth=lineWidths[i],
            linecolor=:grey)
    end
    
    codeWidths *= 8 / maximum(codeWidths)
    labels = map(label->text(label, :top, 8), ena.codeModel[!, :code])
    plot!(p,
        # embedding[1, (codeIndex+1):lineIndex] .- meanX,
        # embedding[2, (codeIndex+1):lineIndex] .- meanY,
        codeXs .- meanX,
        codeYs .- meanY,
        label=nothing,
        seriestype=:scatter,
        series_annotations=labels,
        markershape=:circle,
        markersize=codeWidths,
        markercolor=:white,
        markerstrokecolor=:black
    )

    colors = map(eachrow(ena.metadata[displayRows, :])) do row
        if colormode == :spectral
            return HSL(row[:DayScaled]*240, 1, 0.5)
        elseif colormode == :label
            return colorMap[row[:LABEL]]
        else
            error("Unrecognized color mode $(colormode)")
        end
    end
    
    displayRows2 = [displayRows..., repeat([false], size(embedding, 2) - length(displayRows))...]
    plot!(p,
        embedding[1, displayRows2] .- meanX,
        embedding[2, displayRows2] .- meanY,
        label=nothing,
        seriestype=:scatter,
        markersize=4,
        markercolor=colors,
        markerstrokecolor=colors
    )

    if colormode == :label
        for mylabel in unique(ena.metadata[!, :LABEL])
            color = colorMap[mylabel]
            plot!(p, [-9999], [-9999],
                label=mylabel,
                seriestype=:scatter,
                markersize=4,
                markercolor=color,
                markerstrokecolor=color
            )
        end
    end

    return p
end

## Computing and plotting each desired embedding, ending with the one we'll continue to use below
embedding = nothing
for (weight1, plotsize) in zip(weights1, plotsizes)
    embedding = computeEmbedding(knn, weight1, seed)
    p = plotUMAP(embedding, plotsize, :spectral)
    savefig(p, "images/SpectralUMAP_$(weight1)_$(plotsize).png")
end

# DBSCAN
## Auto-labeling by detected cluster
results = dbscan(embedding[1:2, 1:codeIndex], epsval; min_cluster_size=min_cluster_size, min_neighbors=min_neighbors)
for (i, cluster) in enumerate(results)
    mylabel = "Auto Cluster #$(i)"
    ena.metadata[cluster.core_indices, :LABEL] .= mylabel
    ena.metadata[cluster.boundary_indices, :LABEL] .= mylabel
    if !haskey(colorMap, mylabel)
        if i <= length(EpistemicNetworkAnalysis.DEFAULT_EXTRA_COLORS)
            colorMap[mylabel] = EpistemicNetworkAnalysis.DEFAULT_EXTRA_COLORS[i]
        else
            colorMap[mylabel] = RGB(rand(0.4:0.01:0.9), rand(0.4:0.01:0.9), rand(0.4:0.01:0.9))
        end
    end
end

## Plotting result
p = plotUMAP(embedding, plotsizes[end], :label)
savefig(p, "images/LabelUMAP.png")

## Saving back into dataframe
nonEmptyRows = map(eachrow(data)) do row
    return sum(row[codes]) > 1
end

data[nonEmptyRows, :LABEL] = ena.metadata[!, :LABEL]

## Displaying time windows
for label in sort(unique(data[!, :LABEL]))
    if label != "No Label"
        labelRows = data[!, :LABEL] .== label
        println("$(label): $(first(data[labelRows, :Date])) -- $(last(data[labelRows, :Date]))")
    end
end

# Code and Count
agg_data = combine(groupby(data, :LABEL), sort(codes) .=> sum .=> sort(codes))
display(agg_data)
CSV.write("data/agg_data.csv", agg_data)

# LDA
groups = sort(unique(data[!, :LABEL]))
for dim1 in 1:(length(groups)-3)
    ## Run and plot LDA for all nodes
    rotation = LDARotation(:LABEL, dim1)
    ena = ENAModel(data, codes, conversations, units, dropEmpty=true, rotateBy=rotation, subsetFilter=x->x[:LABEL]!="No Label")
    for lims in limses
        p = plot(ena, weakLinks=false, showUnits=false, lims=lims)    
        savefig(p, "images/LDA$(dim1)-$(lims).png")
    end
end

# MR
for (group1, group2) in MRs
    rotation = MeansRotation(:LABEL, "Auto Cluster #$(group2)", "Auto Cluster #$(group1)")
    ena = ENAModel(data, codes, conversations, units, dropEmpty=true, rotateBy=rotation, subsetFilter=x->x[:LABEL]!="No Label")
    p = plot(ena, weakLinks=false)
    savefig(p, "images/MR_$(group1)_$(group2).png")
    # TODO run mann whitney tests, and pull out and report the coregistrations
end

end # let