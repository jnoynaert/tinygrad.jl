module tinygrad

import Base: +, -, *, /, ^, inv, exp, log, sin, cos, abs

export  +, -, *, /, ^, inv, exp, log, sin, cos, abs,
        Diffable, backwards!, clear!

include("core.jl")

end