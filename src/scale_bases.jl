using ITensors

using ITensors:
  blockdim,
  blockview,
  findfirstblock,
  tensor

# Get the range of the Index `i` for block `b`
# For example, `blockrange(Index([QN(0) => 2, QN(1) => 3]), 2) == 3:5`
function blockrange(i::Index, b::Int)
  start = 1
  for n in 1:b-1
    start += blockdim(i, n)
  end
  stop = start + blockdim(i, b) - 1
  return start:stop
end

"""
   scale_bases(M::MPO;
               normalize_op = "Id",
               combinedind_tags = ITensorLocalMPO.default_combinedind_tags)

Return a change of basis of the MPO that normalizes a specified operator basis element.

# Examples

```julia
using ITensors, ITensorLocalMPO

N = 20
s = siteinds("Electron", N; conserve_qns = true)

U = 1.0
a = AutoMPO()
for n in 1:N-1
  a .+= -1.0, "Cdagup", n, "Cup", n+1
  a .+= -1.0, "Cdagup", n+1, "Cup", n
  a .+= -1.0, "Cdagdn", n, "Cdn", n+1
  a .+= -1.0, "Cdagdn", n+1, "Cdn", n
end
for n in 1:N
  a .+= U, "Nupdn", n
end

C, X, invX = scale_bases(H; normalize_op = "Id")

# Combine the site indices
HX .*= C

# Convert to an MPS
HX = convert(MPS, copy(H))

# Apply the change of basis that scales
# the identity subspace
HX = apply(X, HX)

# Test inverting
Horig = apply(invX, HX)
Horig .*= dag.(C)
Horig = convert(MPO, Horig)

@show norm(H), norm(HX), norm(Horig)
```
"""
function scale_bases(H::MPO; kwargs...)
  s = dag.(firstsiteinds(H; plev = 0))
  return scale_bases(s; kwargs...)
end

default_combinedind_tags(n) = "Comb,Site,n=$n"

function scale_bases(s::Vector{<: Index}; normalize_op = "Id",
                     combinedind_tags = default_combinedind_tags)
  N = length(s)

  ops_to_scale = [op(normalize_op, sn) for sn in s]
  op_norms = norm.(ops_to_scale) .^ 2

  Cs = [combiner(s[n]', dag(s[n]); tags = default_combinedind_tags(n)) for n in 1:N]
  cs = combinedind.(Cs)

  ops_to_scale = ops_to_scale .* Cs

  Xs = [ITensor(cn', dag(cn)) for cn in cs]
  for X in Xs
    dX = minimum(dims(X))
    for n in 1:dX
      X[n, n] = 1
    end
  end
  invXs = deepcopy(Xs)

  # Determine the block of the QN() sector on each site
  qn0blocks = findfirstblock.(indblock -> qn(indblock) == QN(), cs)

  op_vectors = [parent(array(blockview(tensor(ops_to_scale[n]), qn0blocks[n]))) for n in 1:N]
  op_vectors ./= op_norms
  for n in 1:N
    op_vector = op_vectors[n]
    Q_qn0, _ = qr(op_vector)
    U_qn0 = Q_qn0 * I
    U_qn0[:, 1] .= op_vector
    U_qn0 = permutedims(U_qn0)
    range_qn0 = blockrange(cs[n], Int(qn0blocks[n]))
    Xs[n][range_qn0, range_qn0] = U_qn0
    invXs[n][range_qn0, range_qn0] = inv(U_qn0)
  end
  return Cs, Xs, invXs
end

"""
    combine_and_transform(H::MPO, C::Vector{<: ITensor}, X::Vector{<: ITensor}, invX::Vector{<: ITensor})

Combine and transform the sites of the MPO, convert to an MPS.
"""
function combine_and_transform(H::MPO, C::Vector{<: ITensor}, X::Vector{<: ITensor}, invX::Vector{<: ITensor})
  return apply(X, MPS(H .* C))
end

function inv_transform_and_uncombine(HX::MPS, C::Vector{<: ITensor}, X::Vector{<: ITensor}, invX::Vector{<: ITensor})
  return MPO(apply(invX, HX) .* dag.(C))
end

"""
    combine_and_transform(U::Vector{<: ITensor}, H::MPO, C::Vector{<: ITensor}, X::Vector{<: ITensor}, invX::Vector{<: ITensor})

Combine and transform the sites of the MPO as well as the gates. For gates `U`, gates are returned that are the tensor product of `U ⊗ Uᴴ`, with the indices combined and transformed.
"""
function combine_and_transform(U::Vector{<: ITensor}, H::MPO,
                               C::Vector{<: ITensor}, X::Vector{<: ITensor}, invX::Vector{<: ITensor})
  # Combine and transform the sites of the MPO, convert to an MPS
  HX = combine_and_transform(H, C, X, invX)

  UX = Vector{ITensor}(undef, length(U))
  for n in 1:length(U)
    Un = U[n]
    n1, n2 = findsites(H, Un)
    Undag = dag(swapprime(Un, 0 => 1))
    UXn = prime(Un) * prime(Undag, -1)

    C1 = C[n1]
    C2 = C[n2]
    c1 = combinedind(C1)
    c2 = combinedind(C2)

    UXn *= replaceprime(C1, 1 => 2, 0 => -1; inds = !c1)
    UXn *= replaceprime(C2, 1 => 2, 0 => -1; inds = !c2)
    UXn = prime(UXn; inds = (c1, c2))
    UXn *= dag(C1)
    UXn *= dag(C2)

    X1 = X[n1]
    X2 = X[n2]
    invX1 = invX[n1]
    invX2 = invX[n2]

    # TODO: use something like:
    #
    # apply([X1, X2], UXn, [invX1, invX2])
    #
    # for this
    UXn = replaceprime(UXn * prime(X1) * prime(X2), 0 => 1)
    UXn = UXn * invX1 * invX2
    UXn = replaceprime(UXn, 2 => 1)
    UX[n] = UXn
  end
  return UX, HX
end

