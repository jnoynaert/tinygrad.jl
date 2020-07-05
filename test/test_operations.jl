@testset "basic operations" begin

# *, +, sin
x = Diffable(15)
y = Diffable(π/2)
z = x * y + sin(x)

@test backwards!(z) == 1.0
@test backwards!(y) == x.value
@test backwards!(x) == y.value + cos(x.value)

# negation
x = Diffable(4.0)
y = -x

@test backwards!(x) == -1.0

# division

# multiple instances of one variable
x = Diffable(3.0)
y = Diffable(2.0)
z = y * (x * x)
@test backwards!(z) == 1.0
@test backwards!(y) == x.value * x.value
@test backwards!(x) == y.value * 2 * x.value

# cos
x = Diffable(2.0)
z = Diffable(2.0) * cos(x)
@test backwards!(x) == -2.0 * sin(x.value)

# using a function
f(x) = Diffable(3.0) * x
x_1 = Diffable(2.0)
z = f(x_1)
@test backwards!(x_1) == 3.0

# nested functions

# identity

# power

# exponent

# relu

end #basic operations

@testset "interop with base types" begin

x = Diffable(15)
y = π/2
z = x * y + sin(x)

@test backwards!(x) == y + cos(x.value)

x = Diffable(15)
y = π/2
z = x * y + sin(x+2)

@test backwards!(x) == y + cos(x.value+2)

end #interop with base types

@testset "powers & natural log" begin

# x^n alone
a = Diffable(4.0)
y = a^3

@test backwards!(a) == 3 * 4.0^2

# b^x alone

# x^n chained

# b^x chained

# ℯ^x 

x = Diffable(8.9)
y = 3 * exp(x)

@test backwards!(x) == 3 * exp(8.9)

x = Diffable(5.1)
y = exp(x^3)

@test backwards!(x) ≈ exp(5.1^3) * 3 * 5.1^2

# ln
x = Diffable(3.2)
y = log(x)

@test backwards!(x) == 1/3.2

x = Diffable(1.0)
y = exp(log(x))

@test backwards!(x) == 1.0

x = Diffable(-3.2)
@test_throws DomainError log(x) 

end #powers & natural log

