# Nonlinear Projection
## Helper 1 - construct model around the units
function embedUnits!(ena, col, knn, weight1, seed)

    ### Seed
    Random.seed!(seed)

    ### Prepare data
    colVals = Vector{Float64}(ena.metadata[!, col])
    colVals = colVals .- minimum(colVals)
    colVals /= maximum(colVals)
    accumX = transpose(Matrix{Float64}(hcat(colVals, ena.accumModel[!, ena.networkModel[!, :relationship]])))

    ### Model the units
    weightn = (1-weight1) / nrow(ena.networkModel)
    weights = [weight1, [weightn for row in eachrow(ena.networkModel)]...]
    metric = WeightedEuclidean(weights/sum(weights))
    model = UMAP_(accumX; n_neighbors=knn, min_dist=0.0000000001, metric=metric)

    ### Add to accumModel
    ena.accumModel[!, :umap_x] = model.embedding[1, :]
    ena.accumModel[!, :umap_y] = model.embedding[2, :]

    ### Unseed and return model
    Random.seed!(Dates.value(now()))
    return model
end

## Helper 2 - add the networks to the model
function embedNetwork!(ena, model, seed)

    ### Seed
    Random.seed!(seed)

    ### Prepare data
    lineX = nothing
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

    ### Project network components into the embedding
    weightn = 1 / nrow(ena.networkModel)
    weights = [0, [weightn for row in eachrow(ena.networkModel)]...]
    metric = WeightedEuclidean(weights/sum(weights))
    networkEmbedding = UMAP.transform(model, lineX; n_neighbors=1, min_dist=0.0000000001, metric=metric)

    ### Add to networkModel
    ena.networkModel[!, :umap_x] = networkEmbedding[1, :]
    ena.networkModel[!, :umap_y] = networkEmbedding[2, :]

    ### Add to codeModel
    #### Only care about non-empty networks
    nonZeroRows = map(eachrow(ena.accumModel)) do row
        if sum(row[ena.networkModel[!, :relationship]]) == 0
            return false
        else
            return true
        end
    end

    #### Regression model for placing the code dots
    X = Matrix{Float64}(zeros(nrow(ena.accumModel[nonZeroRows, :]), nrow(ena.codeModel)))
    for (i, unitRow) in enumerate(eachrow(ena.accumModel[nonZeroRows, :]))
        for r in keys(ena.relationshipMap)
            a, b = ena.relationshipMap[r]
            X[i, a] += unitRow[r] / 2
            X[i, b] += unitRow[r] / 2
        end
    end

    X = (transpose(X) * X)^-1 * transpose(X)

    #### Running regressions
    meanX = mean(ena.accumModel[nonZeroRows, :umap_x])
    meanY = mean(ena.accumModel[nonZeroRows, :umap_y])
    ena.codeModel[!, :umap_x] = X * Vector{Float64}(ena.accumModel[nonZeroRows, :umap_x] .- meanX)
    ena.codeModel[!, :umap_y] = X * Vector{Float64}(ena.accumModel[nonZeroRows, :umap_y] .- meanY)

    ### Unseed
    Random.seed!(Dates.value(now()))
end

## Helper 3 - plot UMAP+ENA projections
function plotUMAP(ena, colorMap, col; group=nothing, colormode=:spectral, lineSize=8, codeSize=8, unitSize=4)

    ### Empty plot
    p = plot(; size=(800,800))
    xticks!(p, [-1, 1])
    yticks!(p, [-1, 1])
    xlims!(p, (-1.5, 1.5))
    ylims!(p, (-1.5, 1.5))

    ### Hide empty networks, and optionally focus on just one group
    nonZeroRows = map(eachrow(ena.accumModel)) do row
        if sum(row[ena.networkModel[!, :relationship]]) == 0
            return false
        else
            return true
        end
    end

    displayRows = nonZeroRows
    if !isnothing(group)
        displayRows = map(enumerate(eachrow(ena.metadata))) do (i, row)
            if nonZeroRows[i] && row[:LABEL] == group
                return true
            else
                return false
            end
        end
    end

    displayAccums = ena.accumModel[displayRows, :]

    ### Center and scale points
    meanX = mean(ena.accumModel[nonZeroRows, :umap_x])
    meanY = mean(ena.accumModel[nonZeroRows, :umap_y])
    scaleX = 1 / maximum(abs.(ena.accumModel[nonZeroRows, :umap_x] .- meanX))
    scaleY = 1 / maximum(abs.(ena.accumModel[nonZeroRows, :umap_y] .- meanY))
    xs = Vector{Float64}(displayAccums[!, :umap_x])
    xs = xs .- meanX
    xs *= scaleX 
    ys = Vector{Float64}(displayAccums[!, :umap_y])
    ys = ys .- meanY
    ys *= scaleY

    ### Finding network component sizes
    allLineWidths = map(eachrow(ena.networkModel)) do networkRow
        return sum(ena.accumModel[!, networkRow[:relationship]])
    end

    allCodeWidths = zeros(nrow(ena.codeModel))
    for (i, networkRow) in enumerate(eachrow(ena.networkModel))
        j, k = ena.relationshipMap[networkRow[:relationship]]
        allCodeWidths[j] += allLineWidths[i]
        allCodeWidths[k] += allLineWidths[i]
    end

    lineWidths = map(eachrow(ena.networkModel)) do networkRow
        return sum(displayAccums[!, networkRow[:relationship]])
    end

    codeWidths = zeros(nrow(ena.codeModel))
    for (i, networkRow) in enumerate(eachrow(ena.networkModel))
        j, k = ena.relationshipMap[networkRow[:relationship]]
        codeWidths[j] += lineWidths[i]
        codeWidths[k] += lineWidths[i]
    end
    
    ### Finding node positions
    codeXs = ena.codeModel[!, :umap_x] * scaleX
    codeYs = ena.codeModel[!, :umap_y] * scaleY

    ### Draw the network
    lineWidths *= lineSize / maximum(allLineWidths)
    codeWidths *= codeSize / maximum(codeWidths)
    for (i, networkRow) in enumerate(eachrow(ena.networkModel))
        j, k = ena.relationshipMap[networkRow[:relationship]]
        pointA = [codeXs[j], codeYs[j]]
        pointB = [codeXs[k], codeYs[k]]
        pointT = [(networkRow[:umap_x] - meanX) * scaleX, (networkRow[:umap_y] - meanY) * scaleY]
        points = hcat(pointA, pointT, pointT, pointT, pointB)
        plot!(p,
            points[1, :],
            points[2, :],
            label=nothing,
            seriestype=:curves,
            linewidth=lineWidths[i],
            linecolor=:grey)
    end
    
    labels = map(label->text(label, :top, 8), ena.codeModel[!, :code])
    plot!(p,
        codeXs,
        codeYs,
        label=nothing,
        seriestype=:scatter,
        series_annotations=labels,
        markershape=:circle,
        markersize=codeWidths,
        markercolor=:white,
        markerstrokecolor=:black
    )

    ### Draw the units
    colVals = Vector{Float64}(ena.metadata[displayRows, col])
    colVals = colVals .- minimum(colVals)
    colVals /= maximum(colVals)
    colors = map(zip(eachrow(ena.metadata[displayRows, :]), colVals)) do (row, colVal)
        if colormode == :spectral
            return HSL(colVal*240, 1, 0.5)
        elseif colormode == :label
            return colorMap[row[:LABEL]]
        else
            error("Unrecognized color mode $(colormode)")
        end
    end
    
    plot!(p,
        (displayAccums[!, :umap_x] .- meanX) * scaleX,
        (displayAccums[!, :umap_y] .- meanY) * scaleY,
        label=nothing,
        seriestype=:scatter,
        markersize=unitSize,
        markercolor=colors,
        markerstrokecolor=colors
    )

    ### Draw the legend
    if colormode == :label
        if isnothing(group)
            for mylabel in sort(unique(ena.metadata[!, :LABEL]))
                color = colorMap[mylabel]
                plot!(p, [-9999], [-9999],
                    label=mylabel,
                    seriestype=:scatter,
                    markersize=4,
                    markercolor=color,
                    markerstrokecolor=color
                )
            end
        else
            color = colorMap[group]
            plot!(p, [-9999], [-9999],
                label=group,
                seriestype=:scatter,
                markersize=4,
                markercolor=color,
                markerstrokecolor=color
            )
        end
    end

    ### Done, return
    return p
end



# Clustering
## Helper 4 - detect clusters from umap_x and umap_y positions
function autocluster!(ena, data, colorMap, epsval, min_cluster_size, min_neighbors)

    ### Prepare data
    X = Matrix{Float64}(ena.accumModel[!, [:umap_x, :umap_y]])
    X = Matrix{Float64}(transpose(X))
    
    ### Using DBSCAN to detect labels
    results = dbscan(X, epsval; min_cluster_size=min_cluster_size, min_neighbors=min_neighbors)

    ### Applying the labels
    ena.metadata[!, :LABEL] .= "No Label"
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

    ### Saving labels back into original dataframe
    for row in eachrow(ena.metadata)
        dataRows = [join(dataRow[ena.codes], ".") == row[:ENA_UNIT] for dataRow in eachrow(data)]
        data[dataRows, :LABEL] = row[:LABEL]
    end
end



# Descriptive Statistics
## Helper 5 - create df containing upper and lower bound for each labeled group
function findLabelBounds(ena, col)
    labels = String[]
    starts = Any[]
    ends = Any[]
    for label in sort(unique(ena.metadata[!, :LABEL]))
        if label != "No Label"
            labelRows = ena.metadata[!, :LABEL] .== label
            push!(labels, label)
            push!(starts, first(ena.metadata[labelRows, col]))
            push!(ends, last(ena.metadata[labelRows, col]))
        end
    end

    return DataFrame(
        :LABEL => labels,
        :LowerBound => starts,
        :UpperBound => ends
    )
end

## Helper 6 - plot cdfs so we can see how the groups overlap
function plotCDFs(ena, col, colorMap)
    dayLabelMap = Dict(row[col] => row[:LABEL] for row in eachrow(ena.metadata))
    labelCounts = Dict(label => 0 for label in unique(ena.metadata[!, :LABEL]))
    labelPrevX = Dict(label => 0 for label in unique(ena.metadata[!, :LABEL]))
    p = plot(; size=(800,800))
    for (x, day) in enumerate(sort(ena.metadata[!, col]))
        label = dayLabelMap[day]
        if label != "No Label"
            plot!(p,
                [labelPrevX[label], x],
                [labelCounts[label], labelCounts[label]],
                label=nothing,
                seriestype=:line,
                linecolor=colorMap[label])
            
            plot!(p,
                [x, x],
                [labelCounts[label], labelCounts[label] + 1],
                label=nothing,
                seriestype=:line,
                linecolor=colorMap[label])

            labelCounts[label] += 1
            labelPrevX[label] = x
        end
    end

    return p
end



# Preprocessing
## Helper 7 - 
function derivedCode!(data, newCol, oldCols...)
    data[!, newCol] = ones(nrow(data))
    for col in oldCols
        data[!, newCol] = data[!, newCol] .* (1 .- data[!, col])
    end

    data[!, newCol] = 1 .- data[!, newCol]
end