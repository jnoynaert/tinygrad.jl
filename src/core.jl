include("meta.jl")

mutable struct Diffable #<: Real

  value::Float64
  precursors::Vector{Tuple{Float64,Diffable}} #gradient weight + the Diffable node that is dependent during a forward pass of the computation graph (antecedent from a backwards perspective, subsequent from a forward perspective)
  grad

  Diffable(a) = new(a, Vector{Tuple{Float64,Diffable}}(), 0.0)
end

function clear!(x::Diffable)

  x.precursors = similar(x.precursors, 0)
end

"""sadly 8x faster than a cooler one-liner"""
function backwards!(y::Diffable)

    if isempty(y.precursors) # terminal node
      y.grad = 1
    else
      y.grad = 0 # remove this to accumulate
      for (weight,precursor) in y.precursors
        y.grad += weight * backwards!(precursor)
      end
    end

  return y.grad
end


"""
generate conversion functions for a binary operator
e.g. `@create_binaryfns +`
yields
`+(a::Diffable, b) = +(a, Diffable(b))`
`+(a, b::Diffable) = +(Diffable(a), b)`
"""
macro create_conversions(operator)

  quote
    $(esc(operator))(a::Diffable, b::T) where T <: Real = $(esc(operator))(a, Diffable(b))
    $(esc(operator))(a::T, b::Diffable) where T <: Real = $(esc(operator))(Diffable(a), b)
  end
end


"""
generate base functions for a binary operator in terms of arguments `a` and `b`

calculate the result and push the result (weight) and partial derivative into the computation graph

example output:
```
function *(a::Diffable, b::Diffable)

  result = Diffable(a.value * b.value)
  push!(a.precursors, (b.value, result))
  push!(b.precursors, (a.value, result))

  return result
end
```
"""
macro create_binaryfn(operator, ∂a, ∂b, a = :a, b = :b)

  ∂a = replace_expr!(∂a, a => :($a.value), b => :($b.value))
  ∂b = replace_expr!(∂b, a => :($a.value), b => :($b.value))

  quote
    function $(esc(operator))(a::Diffable, b::Diffable)

      result = Diffable($(esc(operator))($a.value, $b.value))
      push!($(:(a.precursors)), ($∂a, result))
      push!($(:(b.precursors)), ($∂b, result))

      return result
    end

    @create_conversions($(esc(operator)))
  end
end


"""unary version"""
macro create_unaryfn(fn, ∂a, a = :a)

  ∂a = replace_expr!(∂a, a => :($a.value))

  quote
    function $(esc(fn))(a::Diffable)

      result = Diffable($(esc(fn))($a.value))
      push!($(:(a.precursors)), ($∂a, result))

      return result
    end
  end

end

# actually define the adjoints:

# binary functions
@create_binaryfn(+, 1.0, 1.0)
@create_binaryfn(-, 1.0, -1.0)
@create_binaryfn(*, b, a)
@create_binaryfn(^, b * a^(b-1), a^b * log(a)) #x^a, a^x

/(a::Diffable, b::Diffable) = a * (b^-1)
@create_conversions /

# unary functions
-(a::Diffable) = -1 * a

@create_unaryfn(inv, -a * a^-2) #used internally for calls to a^-1
@create_unaryfn(exp, result.value) #short-circuit recalculating exp(a)
@create_unaryfn(log, 1/a)
@create_unaryfn(sin, cos(a))
@create_unaryfn(cos, -sin(a))
@create_unaryfn(abs, sign(a)) #equivalent to |x|/x, but handles 0 to avoid random errors at the expense of correctness.

relu(x) = x > 0 ? x : 0
@create_unaryfn(relu, a > 0)
