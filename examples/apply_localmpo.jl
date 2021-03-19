using ITensors,
      ITensorLocalMPO

import ITensors: op

function op(::OpName"Givens", ::SiteType"Electron", s1::Index, s2::Index; θ, θup = θ, θdn = θ)
  ampo = AutoMPO()
  ampo += (θup, "Cdagup", 1, "Cup", 2)
  ampo -= (θup, "Cdagup", 2, "Cup", 1)
  ampo += (θdn, "Cdagdn", 1, "Cdn", 2)
  ampo -= (θdn, "Cdagdn", 2, "Cdn", 1)
  return exp(prod(MPO(ampo, [s1, s2])))
end

function main(N)
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
  H = MPO(a, s)

  θ = π/4
  gates = [("Givens", (n, n+1), (θ = θ,)) for n in 1:N-1]
  U = ops(s, gates)

  # Without basis change
  UH1 = apply(U, H; apply_dag = true, cutoff = 1e-8)

  # With basis change
  basis_transformation = scale_bases(H; normalize_op = "Id")

  UX, HX = combine_and_transform(U, H, basis_transformation...)
  UHX = apply(UX, HX; cutoff = 1e-8)
  UH2 = inv_transform_and_uncombine(UHX, basis_transformation...)
  @show norm(H), norm(HX), norm(UHX), norm(UH1), norm(UH2)
  @show maximum(norm, H), maximum(norm, HX), maximum(norm, UHX), maximum(norm, UH1), maximum(norm, UH2)
  @show inner(UH1, UH2) / (norm(UH1) * norm(UH2))
end

main(100)

