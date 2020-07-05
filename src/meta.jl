# helper functions for defining adjoints
# why be sensible and readable when you can use metaprogramming?

"""symbol or sub-expr replacement in expressions"""
function replace_expr!(e::Expr, pairs...)

  pairs = Dict(pairs)

  if e ∈ keys(pairs)
    return pairs[e]
  end

  for (i, e₀) in enumerate(e.args)
      if e₀ ∈ keys(pairs)
          e.args[i] = pairs[e₀]
      elseif e₀ isa Expr
          replace_expr!(e₀, pairs...)
      end
  end

  return e
end


"""expression case"""
function replace_expr!(e::Symbol, pairs...)

  pairs = Dict(pairs)

  return e ∈ keys(pairs) ? pairs[e] : e
end


"""non-(symbol/expr) cases"""
replace_expr!(e, pairs...) = e