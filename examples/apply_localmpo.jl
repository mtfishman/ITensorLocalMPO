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

function main(N; θ, pattern)
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

  if pattern == "staircase"
    gates = [("Givens", (n, n+1), (θ = θ,)) for n in 1:N-1]
  elseif pattern == "brick"
    gates = [("Givens", (n, n+1), (θ = θ,)) for n in 1:2:N-1]
    append!(gates, [("Givens", (n, n+1), (θ = θ,)) for n in 2:2:N-1])
  else
    error("Gate pattern $pattern not supported")
  end
  U = ops(s, gates)

  # Without basis change
  UH1 = H
  mpo_evolution_succeeded = false
  UH1 = try
    UH1 = apply(U, UH1; apply_dag = true, cutoff = 1e-8)
    UH1 = apply(U, UH1; apply_dag = true, cutoff = 1e-8)
    mpo_evolution_succeeded = true
    UH1
  catch
    println("apply(U, H::MPO, apply_dag = true, [...]) failed, likely because of overflow")
  end

  # With basis change
  basis_transformation = scale_bases(H; normalize_op = "Id")

  UX, HX = combine_and_transform(U, H, basis_transformation...)
  UHX = HX
  UHX = apply(UX, UHX; cutoff = 1e-8)
  UHX = apply(UX, UHX; cutoff = 1e-8)

  # Fix normalization
  normsUHX = norm.(UHX)
  UHX ./= normsUHX
  lognormsUHX = log.(normsUHX)
  lognormUHX = sum(lognormsUHX)
  UHX .*= exp(lognormUHX / N)

  UH2 = inv_transform_and_uncombine(UHX, basis_transformation...)

  @show maxlinkdim(H), maxlinkdim(HX), maxlinkdim(UHX), maxlinkdim(UH2)
  mpo_evolution_succeeded && @show maxlinkdim(UH1)
  @show norm(H), norm(HX), norm(UHX), norm(UH2)
  mpo_evolution_succeeded && @show norm(UH1)
  @show maximum(norm, H), maximum(norm, HX), maximum(norm, UHX), maximum(norm, UH2)
  mpo_evolution_succeeded && @show maximum(norm, UH1)
  mpo_evolution_succeeded && @show inner(UH1, UH2) / (norm(UH1) * norm(UH2))

  return nothing
end

