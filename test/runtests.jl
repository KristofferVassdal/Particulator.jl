using Particulator, Test, StaticArrays, LinearAlgebra, Random
const P = Particulator
const co = P.co
const X0 = SA[0.0, 0.0, 0.0]
const AIR_NEL = co.nair * (0.79 * 14 + 0.21 * 16)
const AIR_I = (0.79 * 82 + 0.21 * 95) * co.eV
const DATA = joinpath(dirname(pathof(Particulator)), "..", "data")
const AIR = Dict("N2" => 0.79 * co.nair, "O2" => 0.21 * co.nair)

# Relativistic state (electron, positron, photon) with kinetic energy `energy` [J].
state(S, species, energy; direction=normalize(SA[1.0, 2.0, 3.0]), t=0.0, w=1.0) =
    S(X0, direction * P.momentum_norm_from_kin(species, energy), w, t)

# Non-relativistic slow electron with kinetic energy `energy` [J] (K = m v^2 / 2).
slowstate(energy; direction=normalize(SA[1.0, 2.0, 3.0]), t=0.0, w=1.0) =
    SlowElectronState(X0, direction * sqrt(2energy / co.electron_mass), w, t)

empty_table() = CollisionTable(proc=(), energy=LinRange(0.0, 1.0, 2), rate=zeros(0, 2), ratebound=0.0)

# Momentum of any particle state (for slow electrons it is m*v).
mom(s) = P.momentum(s)

# Insert a particle without the checks of add_particle! (energy cut, kill-at-birth).
function plant!(popl, s)
    i = popl.n[] + 1
    popl.particles[i] = s
    popl.n[] = i
    P.setr!(popl, i)
    return i
end

# Four-population container with the same layout as scripts/swarm.jl.
function air_populations(n; electron_table=empty_table(), slow_table=empty_table(),
                         photon_table=empty_table(), positron_table=empty_table(),
                         electron_cut=100co.eV, slow_cut=1co.eV, cut=1000co.eV)
    MultiPopulation(:electron => Population(n, ElectronState{Float64}[], electron_table, electron_cut),
                    :slow_electron => Population(n, SlowElectronState{Float64}[], slow_table, slow_cut),
                    :photon => Population(n, PhotonState{Float64}[], photon_table, cut),
                    :positron => Population(n, PositronState{Float64}[], positron_table, cut))
end

include("physics.jl")
include("transport.jl")
