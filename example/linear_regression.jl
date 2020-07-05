using tinygrad

Base.broadcastable(m::Diffable) = Ref(m) #treat diffable as scalar if Diffable not <: Number

function descend(n = 100, α = 1e-1, epochs = 100)

    f(x) = 5x + 4 #true
    y(x) = f(x) + randn() * 0.5 #noised

    ŷ(m,b,x) = m*x + b # estimate

    # MSE loss
    loss(y,ŷ) = abs.(y .- ŷ) .^ 2 |> x-> sum(x) / n # extra abs resolves sign issues on extraneous Diffable promotion

    # make the training data (with pre-normalized inputs)
    xs = range(-1.0; stop = 1.0, length = n) |> collect
    ys = y.(xs)

    m = abs(randn()) |> Diffable
    b = 0 |> Diffable

    ms = Vector{Float64}([m.value])
    bs = Vector{Float64}([b.value])
    
    for epoch in 1:epochs
        ŷ(x) = ŷ(m,b,x)

        ŷs = ŷ.(xs)
        
        currentloss = loss(ys, ŷs)
        backwards!(m)
        backwards!(b)

        m -= m.grad * α; clear!(m)
        b -= b.grad * α; clear!(b)

        push!(ms, m.value)
        push!(bs, b.value)

        epoch % floor(epochs/10) == 0 ? println("Loss on iteration $epoch: $(currentloss.value)") : nothing
    end

    @info "m: $(m.value); b: $(b.value)"
    return ms, bs, xs, ys
end

ms, bs, xs, ys = descend()

using Plots; gr()

# create a scatter plot for each set of values
function create_plot(m,b,iter)

    ŷs = [m * x + b for x in xs]
    plot(xs, ŷs, w = 3)
    scatter!(xs, ys, markerstrokewidth = 0, legend = nothing, title = "Iteration $iter")
end


animation = @animate for i in 1:40
    create_plot(ms[i], bs[i], i)
end

gif(animation, "linear_fit.gif", fps = 3)