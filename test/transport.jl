# Helpers ---------------------------------------------------------------------------------------

# Callback that records the time of every physical collision and checks the time passed to it.
mutable struct Recorder <: AbstractCallback
    times::Vector{Float64}        # state.t at every non-null collision
    passed::Vector{Float64}       # t argument that oncollision received
    lck::ReentrantLock
end
Recorder() = Recorder(Float64[], Float64[], ReentrantLock())
function P.oncollision(c::Recorder, process, state, outcome, t)
    lock(c.lck) do
        process isa NullCollision && return
        push!(c.times, state.t)
        push!(c.passed, t)
    end
    outcome
end

struct TestAbsorption <: P.CollisionProcess end
P.collide(::TestAbsorption, s, energy) = RemoveParticleOutcome(s)

struct LinearPhotonEnergy <: P.AbstractForcing
    pdot::SVector{3,Float64}
end
P.force(f::LinearPhotonEnergy, s::PhotonState) = f.pdot

# Tests -----------------------------------------------------------------------------------------

@testset "Callbacks and population bookkeeping" begin
    cc = CollisionCounter()
    for _ in 1:1000
        P.oncollision(cc, NullCollision(), nothing, nothing, 0.0)
    end
    @test cc.d[NullCollision][] == 1000

    comb = CombinedCallback(VoidCallback(), (cc,))
    P.oncollision(comb, NullCollision(), nothing, nothing, 0.0)
    @test cc.d[NullCollision][] == 1001
    @test P.onstep(comb, nothing, 0.0)

    # repack! moves the active particles to the front and keeps the order of the weights
    pop = Population(10, [state(ElectronState, Electron, 1e6co.eV; w=Float64(i)) for i in 1:6], empty_table())
    remove_particle!(pop, 2); remove_particle!(pop, 5)
    @test nactives(pop) == 4
    repack!(pop)
    @test nparticles(pop) == 4 && nactives(pop) == 4
    @test sort(pop.particles.w[1:4]) == [1.0, 3.0, 4.0, 6.0]
    @test P.weight(pop) == 14.0

    # Population capacity: a full population must refuse more particles.
    full = Population(4, ElectronState{Float64}[], empty_table())
    for _ in 1:4
        @test P.add_particle!(full, state(ElectronState, Electron, 1e6co.eV)) > 0
    end
    @test_throws AssertionError P.add_particle!(full, state(ElectronState, Electron, 1e6co.eV))

    # Particles below the population's energy cut are never added.
    cut = Population(10, ElectronState{Float64}[], empty_table(), 1000co.eV)
    @test P.add_particle!(cut, state(ElectronState, Electron, 999co.eV)) == -1
    @test P.add_particle!(cut, state(ElectronState, Electron, 1001co.eV)) == 1
    @test nactives(cut) == 1
end

@testset "Kill-at-birth of particles below 100 eV (add_particle!)" begin
    # add_particle! keeps particles with K < 100 eV with probability CULL (currently 1e-4) and multiplies
    # their weight by 1/CULL, so the expected weight is conserved.
    CULL = 1e-4
    Random.seed!(11)
    nadd = 2_000_000
    pop = Population(2000, SlowElectronState{Float64}[], empty_table(), 1co.eV)
    for _ in 1:nadd
        P.add_particle!(pop, slowstate(50co.eV; w=1.0))
    end
    n = nactives(pop)
    expected = nadd * CULL                               # Poisson(200)
    @test abs(n - expected) < 6sqrt(expected)
    @test all(w -> w ≈ 1 / CULL, pop.particles.w[1:nparticles(pop)])
    @test P.weight(pop) ≈ nadd rtol=0.5

    # At and above 100 eV nothing is culled.
    pop = Population(1000, SlowElectronState{Float64}[], empty_table(), 1co.eV)
    for _ in 1:1000
        @test P.add_particle!(pop, slowstate(150co.eV; w=2.0)) > 0
    end
    @test all(==(2.0), pop.particles.w[1:1000])

    # Electrons below 100 eV are culled the same way (the cull is energy based, not species based).
    Random.seed!(12)
    pop = Population(2000, ElectronState{Float64}[], empty_table(), 0.0)
    for _ in 1:200_000
        P.add_particle!(pop, state(ElectronState, Electron, 50co.eV))
    end
    @test nactives(pop) <= 20 + 6sqrt(20)                # Poisson(200_000 * CULL)
end

@testset "Roulette and splitting" begin
    Random.seed!(5)
    n = 20_000
    pop = Population(n, [state(ElectronState, Electron, 1e6co.eV) for _ in 1:n], empty_table())
    roulette!(0.25, pop)
    @test abs(nactives(pop) / n - 0.25) < 6sqrt(0.25 * 0.75 / n)
    @test all(p -> p.w == 4.0, eachactive(pop))
    @test abs(P.weight(pop) / n - 1) < 6sqrt(0.25 * 0.75 / n) * 4

    initial = P.setproperties(state(ElectronState, Electron, 1e6co.eV), s=100.0)
    split = Population(3000, [initial], empty_table())
    P.split!(1000, split)
    @test nparticles(split) > 1
    @test abs(P.weight(split) - 1) < 6sqrt(1000) / 1001         # weight conserved on average
    @test all(w -> w ≈ 1 / 1001, split.particles.w[1:nparticles(split)])
    # Every copy inherits the parent's collision clock; independent clocks would be better because
    # the copies otherwise collide at the same time until their first collision.
    clocks = split.particles.s[2:nparticles(split)]
    @test_broken length(unique(clocks)) == length(clocks)
end

@testset "Forces and pushers" begin
    # FieldForcing and ElectromagneticField must give the same force on every particle type.
    efield = HomogeneousField(SA[1e5, -2e5, 3e5])
    bfield = HomogeneousField(SA[0.01, 0.02, -0.03])
    ff = FieldForcing(efield, bfield)
    em = ElectromagneticField(efield, bfield)
    for s in (state(ElectronState, Electron, 1e6co.eV), state(PositronState, Positron, 1e6co.eV),
              slowstate(10co.eV), slowstate(90co.eV; direction=SA[0.0, 0.6, 0.8]))
        @test P.force(ff, s) ≈ P.force(em, s) rtol=1e-12
    end
    # An electron is pushed against E and a positron along E
    s = state(ElectronState, Electron, 1e6co.eV)
    @test P.force(em, ElectronState(X0, X0)) ≈ -co.elementary_charge * SA[1e5, -2e5, 3e5] rtol=1e-12
    @test P.force(em, PositronState(X0, X0)) ≈ co.elementary_charge * SA[1e5, -2e5, 3e5] rtol=1e-12
    @test iszero(P.force(em, state(PhotonState, Photon, 1e6co.eV)))
    @test iszero(P.force(NullForcing(), s))

    # Free flight
    s = state(ElectronState, Electron, 1e6co.eV)
    free = P.advance_particle(RK2Pusher(NullForcing()), s, 1e-9)
    @test free.x ≈ s.x + P.velocity(s) * 1e-9 rtol=1e-12
    @test free.p == s.p
    @test free.t == 1e-9

    # Constant electric field: RK2 integrates uniform acceleration exactly.
    E = SA[0.0, 0.0, -3e5]
    uniform = FieldForcing(HomogeneousField(E), HomogeneousField(X0))
    s = state(ElectronState, Electron, 1e6co.eV; direction=SA[0.0, 0.0, 1.0])
    f = s
    for _ in 1:10
        f = P.advance_particle(RK2Pusher(uniform), f, 1e-11)
    end
    @test f.p ≈ s.p + 1e-10 * P.force(uniform, s) rtol=1e-12
    @test kinenergy(f) > kinenergy(s)               # electrons run away along +z for E along -z

    v0 = SA[1e6, 0.0, 2e6]
    ss = SlowElectronState(X0, v0)
    f = ss
    for _ in 1:10
        f = P.advance_particle(RK2Pusher(uniform), f, 1e-11)
    end
    a = -co.elementary_charge * E / co.electron_mass
    @test f.v ≈ v0 + a * 1e-10 rtol=1e-12
    @test f.x ≈ v0 * 1e-10 + a * (1e-10)^2 / 2 rtol=1e-10
    @test f.t ≈ 1e-10 rtol=1e-12

    # Gyration in a magnetic field: radius p/(eB), period 2πγm/(eB), second order convergence.
    B = 0.1
    mag = FieldForcing(HomogeneousField(X0), HomogeneousField(SA[0.0, 0.0, B]))
    s = state(ElectronState, Electron, 1e6co.eV; direction=SA[1.0, 0.0, 0.0])
    period = 2π * P.gamma(s) * co.electron_mass / (co.elementary_charge * B)
    radius = norm(s.p) / (co.elementary_charge * B)
    errors = Float64[]; drifts = Float64[]
    for N in (100, 200, 400)
        f = s
        for _ in 1:N
            f = P.advance_particle(RK2Pusher(mag), f, period / N)
        end
        push!(errors, norm(f.x - s.x) / radius)
        push!(drifts, abs(kinenergy(f) / kinenergy(s) - 1))
    end
    @test errors[3] < errors[2] < errors[1]
    @test errors[2] / errors[3] > 3                     # second order
    @test errors[3] < 1e-3
    @test drifts[3] < 1e-3
    # The magnetic force is perpendicular to the velocity: for one step energy changes only at O(Δt³)
    f = P.advance_particle(RK2Pusher(mag), s, period / 1e4)
    @test abs(kinenergy(f) / kinenergy(s) - 1) < 1e-9

    # RestrictedForcing/RestrictedPusher act on one species only
    r = RestrictedForcing{PositronState}(uniform)
    @test iszero(P.force(r, state(ElectronState, Electron, 1e6co.eV)))
    @test P.force(r, state(PositronState, Positron, 1e6co.eV)) ≈ P.force(uniform, state(PositronState, Positron, 1e6co.eV))
    rp = RestrictedPusher{PositronState{Float64}}(RK2Pusher(uniform))
    e = state(ElectronState, Electron, 1e6co.eV)
    moved = P.advance_particle(rp, e, 1e-9)
    @test moved.p == e.p && moved.x == e.x && moved.t == 1e-9
end

@testset "Manufactured varying-rate survival law" begin
    # A manufactured photon forcing gives K(t)=K0*(1+t/tfinal), so nu(t)=1e9*(1+t/tfinal).
    # This tests thinning against exp(-integral(nu dt)), not a physical photon force.
    energy = 1e6co.eV
    tfinal = 1e-9
    # One coarse cell deliberately makes the majorant loose, exposing missed thinning.
    grid = LinRange(0.5energy, 2.5energy, 2)
    ct = CollisionTable(proc=(TestAbsorption(),), energy=grid,
                        rate=reshape(1e9 .* grid ./ energy, 1, :), ratebound=2.5e9)
    pusher = RK2Pusher(LinearPhotonEnergy(SA[energy / (co.c * tfinal), 0.0, 0.0]))
    for (dt, n) in ((1e-10, 100_000), (2.5e-11, 10_000))
        Random.seed!(123)
        mp = MultiPopulation(:photon => Population(n, [state(PhotonState, Photon, energy; direction=SA[1.0, 0.0, 0.0]) for _ in 1:n], ct))
        cb = Recorder()
        for k in 1:round(Int, tfinal / dt)
            advance!(mp, pusher, k * dt, cb)
        end
        times = sort(cb.times)
        for fraction in (0.25, 0.5, 1.0)
            expected = 1 - exp(-(fraction + fraction^2 / 2))
            observed = count(t -> t <= fraction * tfinal * (1 + 1e-9), times) / n
            @test abs(observed - expected) < 6sqrt(expected * (1 - expected) / n)
        end
        survival = exp(-1.5)
        @test abs(nactives(get(mp, Photon)) / n - survival) < 6sqrt(survival * (1 - survival) / n)
        distance = abs(length(times) / n - (1 - survival))
        for (i, t) in enumerate(times)
            expected = 1 - exp(-(t / tfinal + (t / tfinal)^2 / 2))
            distance = max(distance, abs(i / n - expected), abs((i - 1) / n - expected))
        end
        @test sqrt(n) * distance < 2.8                          # Kolmogorov-Smirnov
        @test all(p -> p.t ≈ tfinal, eachactive(get(mp, Photon)))

        # oncollision must receive the time of the collision, i.e. state.t.
        @test_broken all(cb.passed .≈ cb.times)
    end
end

@testset "Electron/slow-electron handoff" begin
    cut = 1000co.eV
    # Electrons at or below `cut` become slow electrons; slow electrons above it become electrons.
    mp = air_populations(100; electron_cut=cut)
    electrons, slow = get(mp, Electron), get(mp, SlowElectron)
    # (plant! inserts directly: add_particle! refuses particles below the population's own cut)
    plant!(electrons, state(ElectronState, Electron, 500co.eV; w=2.0, t=3e-9))
    P.add_particle!(electrons, state(ElectronState, Electron, 5000co.eV; w=3.0))
    P.ElectronStateToSlowElectronState!(mp)
    @test nactives(electrons) == 1 && nactives(slow) == 1
    s = first(eachactive(slow))
    @test s.w == 2.0
    @test kinenergy(s) ≈ 500co.eV rtol=5e-3          # ½mv² vs relativistic K: O(K/mc²)
    @test norm(s.v) < 0.2co.c
    @test normalize(s.v) ≈ normalize(SA[1.0, 2.0, 3.0]) rtol=1e-12
    # The particle clock must survive the handoff; the conversion currently passes `s` where the
    # constructor expects `t`, so t becomes the O(1) collision clock.
    @test s.t == 3e-9

    P.add_particle!(slow, slowstate(2000co.eV; w=5.0, t=4e-9))
    P.SlowElectronStateToElectronState!(mp)
    @test nactives(slow) == 1                        # the 500 eV one stays slow, 2 keV was converted
    @test nactives(electrons) == 2
    converted = [p for p in eachactive(electrons) if p.w == 5.0]
    @test length(converted) == 1
    c = P.instantiate(only(converted))
    @test kinenergy(c) ≈ 2000co.eV rtol=1e-12        # built from the slow state's own energy
    @test normalize(c.p) ≈ normalize(SA[1.0, 2.0, 3.0]) rtol=1e-12
    @test c.t == 4e-9

    # Round trip changes the energy only at the ½mv² vs K level
    mp = air_populations(100; electron_cut=cut)
    plant!(get(mp, Electron), state(ElectronState, Electron, 900co.eV))
    P.ElectronStateToSlowElectronState!(mp)
    s = P.instantiate(first(eachactive(get(mp, SlowElectron))))
    @test kinenergy(s) < 900co.eV                    # ½mv² underestimates K slightly
    @test kinenergy(s) ≈ 900co.eV rtol=5e-3

    # droplow! removes particles below each population's own cut; positrons annihilate into photons.
    mp = air_populations(100; electron_cut=cut)
    @test P.add_particle!(get(mp, SlowElectron), slowstate(0.5co.eV)) == -1     # refused by the 1 eV cut
    plant!(get(mp, SlowElectron), slowstate(0.5co.eV))
    plant!(get(mp, Positron), state(PositronState, Positron, 500co.eV))
    plant!(get(mp, Photon), state(PhotonState, Photon, 10co.eV))
    P.droplow!(mp)
    @test nactives(get(mp, SlowElectron)) == 0
    @test nactives(get(mp, Positron)) == 0
    photons = collect(P.instantiate(p) for p in eachactive(get(mp, Photon)))
    @test length(photons) == 2
    @test all(p -> isapprox(kinenergy(p), co.electron_mc2; rtol=1e-12), photons)
end

@testset "Slow electrons keep being advanced after a collision" begin
    # Elastic collisions at a very high rate: the particle must keep colliding for the whole run.
    Random.seed!(3)
    grid = LinRange(0.0, 1e-17, 2)
    table = CollisionTable(proc=(Elastic(1e-4),), energy=grid, rate=fill(1e12, 1, 2), ratebound=1e12)
    pop = Population(10, [slowstate(10co.eV)], table, 1co.eV)
    mp = MultiPopulation(:slow_electron => pop)
    cc = CollisionCounter()
    for k in 1:10
        advance!(mp, RK2Pusher(NullForcing()), k * 1e-10, cc)
    end
    ncoll = get(cc.d, Elastic{Float64}, Threads.Atomic{Int}(0))[]
    @test ncoll > 1                                  # at least one collision happened
    @test ncoll > 500                         # about 1e12 * 1e-9 = 1000 expected
    @test pop.particles.t[1] ≈ 1e-9           # the clock jumps to O(1) s after a collision
end

@testset "Air collision tables and rate bounds" begin
    n2, o2 = AIR["N2"], AIR["O2"]
    cut = 1000co.eV
    Fdt = co.elementary_charge * 3.5e5 * 25e-12
    electrons = [(2n2, RelativisticCoulomb(7)), (2o2, RelativisticCoulomb(8)),
                 (2n2, SeltzerBerger(7)), (2o2, SeltzerBerger(8)),
                 [(n2, p) for p in ORBITALS["N2"]]..., [(o2, p) for p in ORBITALS["O2"]]...]
    positrons = [(2n2, RelativisticCoulomb(7)), (2o2, RelativisticCoulomb(8)),
                 (2n2, Bhaba(7, cut)), (2o2, Bhaba(8, cut)),
                 (2n2, PositronAnihilation(7)), (2o2, PositronAnihilation(8))]
    photons = [(2n2, PhotoElectric(7)), (2o2, PhotoElectric(8)),
               (2n2, Compton(7)), (2o2, Compton(8)), (2n2, BetheHeitler(7)), (2o2, BetheHeitler(8))]
    ec = collision_table_from_processes(electrons, Electron, Fdt; safety=1.15)
    pc = collision_table_from_processes(positrons, Positron, Fdt; safety=1.15)
    gc = collision_table_from_processes(photons, Photon, 0.0; safety=1.15)
    for (name, table) in (("electron", ec), ("positron", pc), ("photon", gc))
        @test length(table) == (name == "electron" ? length(electrons) : 6)
        for energy in 10 .^ range(log10(2e3), 8.4; length=400) .* co.eV
            pre = P.presample(table, nothing, energy)
            rates = [P.rate(table, j, pre) for j in 1:length(table)]
            total = sum(rates)
            # Rates are fits and can be slightly negative next to thresholds, never by much.
            @test all(>(-1e-3 * max(total, 1.0)), rates)
            @test total <= P.ratebound(table, energy, pre) * (1 + 1e-3) + 1e-6
        end
    end

    # Rate bound also from 1 keV down to the cut for electrons, with no Fdt: bound is the rate.
    ec0 = collision_table_from_processes(electrons, Electron, 0.0; safety=1.0)
    for energy in (200, 1e3, 3e3, 1e5, 1e7) .* co.eV
        pre = P.presample(ec0, nothing, energy)
        total = sum(P.rate(ec0, j, pre) for j in 1:length(ec0))
        @test total <= P.ratebound(ec0, energy, pre) * (1 + 1e-12)
    end
end

@testset "Coupled ensemble: run! smoke test with all four populations" begin
    n2, o2 = AIR["N2"], AIR["O2"]
    cut = 1000co.eV
    dt = 25e-12
    Fdt = co.elementary_charge * 3e5 * dt
    ecolls = collision_table_from_processes(
        [(2n2, RelativisticCoulomb(7)), (2o2, RelativisticCoulomb(8)),
         (2n2, SeltzerBerger(7)), (2o2, SeltzerBerger(8)),
         [(n2, p) for p in ORBITALS["N2"]]..., [(o2, p) for p in ORBITALS["O2"]]...],
        Electron, Fdt; safety=1.15)
    pcolls = collision_table_from_processes(
        [(2n2, RelativisticCoulomb(7)), (2o2, RelativisticCoulomb(8)),
         (2n2, Bhaba(7, cut)), (2o2, Bhaba(8, cut)),
         (2n2, PositronAnihilation(7)), (2o2, PositronAnihilation(8))],
        Positron, Fdt; safety=1.15)
    gcolls = collision_table_from_processes(
        [(2n2, PhotoElectric(7)), (2o2, PhotoElectric(8)), (2n2, Compton(7)), (2o2, Compton(8)),
         (2n2, BetheHeitler(7)), (2o2, BetheHeitler(8))], Photon, 0.0; safety=1.15)
    cross = open(io -> parse_bolsigplus(io), joinpath(DATA, "Cross_section_N2_O2.txt"))
    lcolls = collision_table_from_lxcat(cross, SlowElectron, Fdt; safety=1.15)

    Random.seed!(2024)
    es = [state(ElectronState, Electron, (1e5 + 5e6rand()) * co.eV; direction=P.randsphere()) for _ in 1:20]
    mp = MultiPopulation(:electron => Population(100_000, es, ecolls, 100co.eV),
                         :slow_electron => Population(100_000, SlowElectronState{Float64}[], lcolls, 1co.eV, 100co.eV),
                         :photon => Population(100_000, PhotonState{Float64}[], gcolls, cut),
                         :positron => Population(100_000, PositronState{Float64}[], pcolls, cut))
    init!(mp)
    loss = ChebContinuumLoss(ContinuumLoss(AIR_NEL, AIR_I, cut), 1e9co.eV, 3)
    field = ElectromagneticField(HomogeneousField(SA[0.0, 0.0, -3e5]), HomogeneousField(X0))
    pusher = RK2Pusher(CombinedForcing(field, RestrictedForcing{PositronState}(loss)))
    cc = CollisionCounter()
    run!(mp, pusher, 1e-9, dt, cc; output_dt=nothing, verbosity=0)

    # Everything remains finite, weights stay positive and each population respects its cut.
    for (name, popl) in pairs(mp)
        for p in eachactive(popl)
            @test all(isfinite, p.x) && isfinite(p.w) && p.w > 0 && isfinite(kinenergy(p))
        end
    end
    @test nactives(get(mp, Electron)) >= 20
    @test all(kinenergy(p) >= 100co.eV for p in eachactive(get(mp, Electron)))
    @test all(kinenergy(p) >= 1co.eV for p in eachactive(get(mp, SlowElectron)))
    @test all(p.t ≈ 1e-9 for p in eachactive(get(mp, Electron)))
    @test sum(x -> x[][], values(cc.d)) > 0
end
