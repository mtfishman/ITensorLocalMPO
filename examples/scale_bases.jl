using ITensors
using ITensorLocalMPO

function fermion_hopping_mpo(N::Int)
  s = siteinds("Fermion", N; conserve_qns = true)
  U = 1.0
  a = AutoMPO()
  for n in 1:N-1
    a .+= -1.0, "Cdag", n, "C", n+1
    a .+= -1.0, "Cdag", n+1, "C", n
  end
  for n in 1:N
    a .+= U, "N", n
  end
  return MPO(a, s)
end

function electron_hopping_mpo(N::Int)
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
  return MPO(a, s)
end

function hopping_mpo(st::String, N::Int)
  return if st == "Electron"
    electron_hopping_mpo(N)
  elseif st == "Fermion"
    fermion_hopping_mpo(N)
  else
    error("Site type $st not supported")
  end
end

function id_mpo(st::String, N::Int)
  s = siteinds(st, N; conserve_qns = true)
  return MPO(s, "Id")
end

function main(st::String, N::Int; mpo = "Id")
  H = if mpo == "Id"
    id_mpo(st, N)
  elseif mpo == "Hopping"
    hopping_mpo(st, N)
  else
    error("MPO type $mpo not supported right now")
  end
  C, X, invX = scale_bases(H; normalize_op = "Id")
  HX = convert(MPS, copy(H))
  HX .*= C
  HX = apply(X, HX)

  # Test inverting
  Horig = apply(invX, HX)
  Horig .*= dag.(C)
  Horig = convert(MPO, Horig)

  println("Site type: $st")
  println("Number of sites: $N")
  println("MPO type: $mpo")
  @show norm(H), norm(HX), norm(Horig)
  println()
  return HX
end

N = 20
main("Fermion", N; mpo = "Id")
main("Fermion", N; mpo = "Hopping")
main("Electron", N; mpo = "Id")
main("Electron", N; mpo = "Hopping")

nothing
