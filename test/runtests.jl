using Test
using tinygrad

compare_Zygote = false

include("test_operations.jl")

if compare_Zygote
    using Zygote
    include("test_integration.jl")
end