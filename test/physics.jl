@testset "Rotation and collision kinematics" begin
    positron = PositronState(X0, SA[1e-22, 0.0, 0.0])
    @test positron.t == 0 && positron.w == 1 && positron.active && positron.s > 0
    electron = ElectronState(X0, SA[1e-22, 0.0, 0.0])
    @test P.momentum_from_v(Electron, SA[1e7, 0.0, 0.0]) ≈ SA[1e7, 0.0, 0.0] * co.electron_mass rtol=1e-3
    @test P.momentum_norm_from_kin(Electron, 1e6co.eV) ≈ norm(P.momentum(state(ElectronState, Electron, 1e6co.eV))) rtol=1e-13

    # turn(u, cosθ, ϕ, n) must return a vector of norm n at an angle θ from u.
    # Axes within ~1e-7 of ±z are excluded, see the @test_broken block below.
    axes_ok = (SA[1.0, 0.0, 0.0], SA[0.0, 1.0, 0.0], normalize(SA[1.0, 2.0, 3.0]),
               normalize(SA[1e-3, 0.0, 1.0]), normalize(SA[1e-3, 0.0, -1.0]))
    for axis in axes_ok, cosine in (-1.0, 0.0, 0.5, 1.0)
        v = P.turn(axis, cosine, 0.7, 3.0)
        @test all(isfinite, v)
        @test norm(v) ≈ 3.0 rtol=1e-10
        @test dot(axis, v) / 3 ≈ cosine atol=1e-10
    end

    # KNOWN LIMITATION of util.jl: turn() divides by sqrt(1 - μz^2), so it is NaN for exact ±z
    # and inaccurate for tilts below ~1e-7. Flip these to @test once turn() is pole-safe.
    for axis in (SA[0.0, 0.0, 1.0], SA[0.0, 0.0, -1.0])
        v = P.turn(axis, 0.5, 0.7, 3.0)
        @test_broken !any(isnan, v)
    end

    Random.seed!(42)
    directions = (normalize(SA[1.0, 2.0, 3.0]), normalize(SA[1e-3, 0.0, 1.0]))
    for ev in (3e3, 1e4, 1e6, 1e8), direction in directions
        for (process, S, species) in ((Moller(7, 1000co.eV), ElectronState, Electron),
                                      (Bhaba(7, 1000co.eV), PositronState, Positron))
            s = state(S, species, ev * co.eV; direction)
            closure = Bool[]
            for _ in 1:50
                o = P.collide(process, s, kinenergy(s))
                @test kinenergy(o.state1) + kinenergy(o.state2) ≈ kinenergy(s) rtol=1e-10
                push!(closure, norm(o.state1.p + o.state2.p - s.p) <= 1e-10 * norm(s.p))
            end
            # KNOWN BUG: the two secondaries are given azimuths ϕ and -ϕ, but momentum conservation
            # needs ϕ and ϕ+π, so momentum is not conserved (residual ~60 % of p at 3 keV, ~1e-3 at
            # 100 MeV). Energy is conserved. Flip to @test once moller.jl/bhaba.jl/rbeb.jl are fixed.
            @test_broken all(closure)
        end

        # RBEB ionisation. The secondary is an ElectronState above 100 eV and a SlowElectronState
        # (kinetic energy ½mv²) below, so only the generic `momentum`/`kinenergy` are used.
        opposed = Bool[]
        for mol in ("N2", "O2"), orb in ORBITALS[mol]
            s = state(ElectronState, Electron, ev * co.eV; direction)
            o = P.collide(orb, s, kinenergy(s))
            @test o.state1 isa ElectronState
            @test o.state2 isa Union{ElectronState,SlowElectronState}
            @test all(isfinite, mom(o.state1)) && all(isfinite, mom(o.state2))
            # Slow states use the non-relativistic ½mv², which differs from K by O(K/mc²) ≲ 1e-4.
            @test kinenergy(o.state1) + kinenergy(o.state2) + orb.B ≈ kinenergy(s) rtol=1e-4
            if o.state2 isa SlowElectronState
                @test kinenergy(o.state2) <= 1.001 * 100co.eV
            else
                @test kinenergy(o.state2) > 99co.eV
            end
            # Bound targets absorb momentum: test opposing transverse azimuths, not free closure.
            a = mom(o.state1) - dot(mom(o.state1), direction) * direction
            b = mom(o.state2) - dot(mom(o.state2), direction) * direction
            push!(opposed, dot(a, b) <= 0)
        end
        # Same ϕ / -ϕ azimuth problem as for Moller and Bhabha: the transverse momenta of the two
        # outgoing electrons are not opposite.
        @test_broken all(opposed)
    end

    # Below the binding energy RBEB must do nothing.
    orb = ORBITALS["N2"][end]
    s = state(ElectronState, Electron, 0.5 * orb.B)
    @test P.collide(orb, s, kinenergy(s)) isa P.NullOutcome

    # Pair production: e- + e+ share the photon energy minus 2 m c².
    s = state(PhotonState, Photon, 10e6co.eV)
    o = P.collide(BetheHeitler(7), s, kinenergy(s))
    @test o isa ReplaceParticlePairOutcome
    @test o.state2 isa ElectronState && o.state3 isa PositronState
    @test kinenergy(o.state2) + kinenergy(o.state3) + 2co.electron_mc2 ≈ kinenergy(s) rtol=1e-12
    u = normalize(s.p)
    a = o.state2.p - dot(o.state2.p, u) * u
    b = o.state3.p - dot(o.state3.p, u) * u
    # KNOWN BUG: electron and positron are rotated by the same azimuth ϕ (Geant4 uses ϕ and ϕ+π), so
    # their transverse momenta point the same way instead of cancelling.
    @test_broken dot(a, b) < 0

    # Annihilation in flight
    for ev in (1e3, 1e6, 1e8)
        pos = state(PositronState, Positron, ev * co.eV)
        o = P.collide(PositronAnihilation(7), pos, kinenergy(pos))
        @test o.state2 isa PhotonState && o.state3 isa PhotonState
        @test kinenergy(o.state2) + kinenergy(o.state3) ≈ kinenergy(pos) + 2co.electron_mc2 rtol=1e-11
        @test norm(o.state2.p + o.state3.p - pos.p) <= 1e-11 * norm(pos.p)
    end

    # Positron drop: two back-to-back 511 keV photons
    pos = state(PositronState, Positron, 500co.eV; w=3.0, t=1e-12)
    o = P.drop(pos)
    @test o isa ReplaceParticlePairOutcome
    @test o.state2.w == 3 && o.state3.w == 3 && o.state2.t == 1e-12
    @test kinenergy(o.state2) ≈ co.electron_mc2 rtol=1e-12
    @test norm(o.state2.p + o.state3.p) <= 1e-14 * norm(o.state2.p)
end

@testset "Bremsstrahlung" begin
    sb = SeltzerBerger(7)
    @test totalcs(sb, exp(first(sb.log_energy))) ≈ first(sb.totalcs) rtol=1e-10
    # (exactly at the last grid energy totalcs indexes one past the end; stay just below it)
    @test totalcs(sb, exp(last(sb.log_energy)) * 0.9999) ≈ last(sb.totalcs) rtol=1e-3
    @test all(>(0), sb.totalcs)
    @test issorted(sb.pcum) && first(sb.pcum) == 0 && last(sb.pcum) == 1

    # The sampled secondary energy is a fraction of the primary energy.
    Random.seed!(42)
    for j in (5, 20, 40, length(sb.log_energy) - 1)
        energy = exp(sb.log_energy[j]) * 1.001
        fractions = [P.sample_secondary_energy(sb, energy) for _ in 1:2000]
        @test all(x -> 0 < x < 1, fractions)
    end

    # (energies that sit exactly on a table node are avoided: there the sampled fraction can be
    # exactly 1 and collide() trips its `k < eng` assertion with probability ~1e-3)
    for ev in (1.3e4, 1.3e5, 1.3e6, 7e7)
        s = state(ElectronState, Electron, ev * co.eV; direction=normalize(SA[1.0, 2.0, 3.0]))
        balance = Float64[]
        for _ in 1:500
            o = P.collide(sb, s, kinenergy(s))
            @test o.state1 isa ElectronState && o.state2 isa PhotonState
            @test 0 < kinenergy(o.state2) < kinenergy(s)
            @test all(isfinite, o.state1.p)
            push!(balance, (kinenergy(o.state1) + kinenergy(o.state2)) / kinenergy(s) - 1)
        end
        # KNOWN LIMITATION: the electron momentum is p0 - p_photon, which does not conserve energy:
        # E_e + k exceeds E0 by ~ k(1-β) on average (mean +6 % of E0 at 100 keV, +35 % at 1 keV,
        # negligible above ~10 MeV). Flip to @test when collide(::SeltzerBerger) sets E_e = E0 - k.
        @test_broken maximum(abs, balance) < 1e-6
        @test maximum(balance) < 1.5        # ... but never absurd
    end
end

@testset "Moller and Bhabha secondary spectra" begin
    Random.seed!(123)
    nsamples = 40_000
    for (ev, fraction) in ((1e6, 0.001), (1e6, 0.1), (50e6, 0.3))
        energy = ev * co.eV
        gamma = 1 + energy / co.electron_mc2
        beta2 = 1 - 1 / gamma^2

        # Moller: dσ/dε ∝ 1/ε² ... (Geant4 PRM 11.3), 1/(ε(1-ε)) terms; compare with the totalcs
        # and bounds only.
        samples = [P.sample_secondary_energy(Moller(7, fraction * energy), energy) / energy
                   for _ in 1:nsamples]
        @test all(x -> fraction <= x <= 0.5 + 1e-12, samples)

        # Bhabha: pdf ∝ 1/(β²ε²) - B1/ε + B2 - B3 ε + B4 ε²  (this is also the integrand of totalcs)
        b1, b2, b3, b4 = P.bhaba_bs(1 / (gamma + 1))
        integral(x) = -1 / (beta2 * x) - b1 * log(x) + b2 * x - b3 * x^2 / 2 + b4 * x^3 / 3
        moment(x) = log(x) / beta2 - b1 * x + b2 * x^2 / 2 - b3 * x^3 / 3 + b4 * x^4 / 4
        expected = (moment(1) - moment(fraction)) / (integral(1) - integral(fraction))
        process = Bhaba(7, fraction * energy)
        samples = [P.sample_secondary_energy(process, energy) / energy for _ in 1:nsamples]
        average = sum(samples) / nsamples
        sem = sqrt(sum(abs2, samples .- average) / (nsamples - 1) / nsamples)
        @test all(x -> fraction <= x <= 1, samples)
        ok = abs(average - expected) < 6sem
        for x in (fraction + (1 - fraction) / 4, fraction + (1 - fraction) / 2)
            expectedcdf = (integral(x) - integral(fraction)) / (integral(1) - integral(fraction))
            ok &= abs(count(<=(x), samples) / nsamples - expectedcdf) < 6 / sqrt(nsamples)
        end
        # KNOWN BUG: the acceptance function g in bhaba.jl uses +B1 ε and +B3 ε³, but the Bhabha
        # pdf (and the integrand of its own totalcs) has -B1 ε and -B3 ε³ (Geant4 PRM 11.3).
        # The sample mean/CDF are therefore off. Flip to @test once the signs are fixed.
        @test_broken ok
    end
end

@testset "Restricted and total stopping" begin
    for ev in (2e3, 1e5, 1e6, 1e7), fraction in (0.001, 0.1, 0.5, 1.0)
        energy = ev * co.eV
        cl = ContinuumLoss(AIR_NEL, AIR_I, fraction * energy)
        for S in (ElectronState, PositronState)
            loss = P.energy_loss(cl, S, energy)
            @test isfinite(loss)
            # (the positron expression is wrong for large cuts, see the _F test below)
            (S == ElectronState || fraction <= 0.1) && @test loss > 0
        end
    end

    # Restricted loss grows with the cut and the density correction δ lowers it at high energy.
    for ev in (1e5, 1e7), S in (ElectronState, PositronState)
        energy = ev * co.eV
        l1 = P.energy_loss(ContinuumLoss(AIR_NEL, AIR_I, 0.001energy), S, energy)
        l2 = P.energy_loss(ContinuumLoss(AIR_NEL, AIR_I, 0.1energy), S, energy)
        # (positrons: broken by the _F typo described below, the restricted loss then even drops)
        S == ElectronState ? (@test l1 < l2) : (@test_skip l1 < l2)
    end

    # Restricted loss of a 1 MeV electron (cut 100 keV) in air: ~0.2 MeV/m (NIST ESTAR collision
    # stopping power ~1.9 MeV cm²/g times 1.2e-3 g/cm³, a little less when restricted).
    cl = ContinuumLoss(AIR_NEL, AIR_I, 1e5co.eV)
    @test 0.15e6 < P.energy_loss(cl, ElectronState, 1e6co.eV) / co.eV < 0.25e6

    # Positron expression vs. Geant4 PRM 11.3: F+ = ln(τ τup) - β²/τ [τ + 2τup - y(3τup²/2 + y(τup - τup³/3
    # + y(τup²/2 - τ τup³/3 + τup⁴/4)))]. For τup = τ it reduces to the Rohrlich-Carlson unrestricted
    # term 2ln τ - β²/12 (23 + 14y + 10y² + 4y³).
    fplus(τ, d, β2) = (y = 1 / (τ + 2); d2 = d^2 / 2; d3 = d^3 / 3; d4 = d^4 / 4;
                       log(τ * d) - β2 * (τ + 2d - y * (3d2 + y * (d - d3 + y * (d2 - τ * d3 + d4)))) / τ)
    okF = true
    for ev in (1e5, 1e6, 1e7), fraction in (0.001, 0.1, 0.5, 1.0)
        τ = ev * co.eV / co.electron_mc2; β2 = 1 - 1 / (1 + τ)^2
        okF &= isapprox(P._F(PositronState, τ, fraction * τ, β2), fplus(τ, fraction * τ, β2); rtol=1e-12)
    end
    # KNOWN BUG in continuum.jl `_F(::Type{<:PositronState}, ...)`: the prefactor is τup²/τ instead of
    # β²/τ and the bracket starts with τ*2τup instead of τ + 2τup. The error is ~1 in a bracket of
    # ~20 (4-5 % in the positron friction) and grows for large cuts. Flip to @test once fixed.
    @test_broken okF
    for ev in (1e5, 1e6, 1e7)
        τ = ev * co.eV / co.electron_mc2; γ = 1 + τ; β2 = 1 - 1 / γ^2; y = 1 / (γ + 1)
        @test fplus(τ, τ, β2) ≈ 2log(τ) - β2 / 12 * (23 + 14y + 10y^2 + 4y^3) rtol=1e-12
    end

    # Density-effect correction δ at x = log10(βγ): zero below x0, 2 ln10 x - C above x1 (C is
    # determined by the electron density), smooth in between. The two middle values are regression
    # values of the current implementation.
    cl = ContinuumLoss(AIR_NEL, AIR_I, 1000co.eV)
    hνp = co.hbar * co.c * sqrt(4π * AIR_NEL * co.r_e)
    C = 1 + 2log(AIR_I / hνp)
    for (x, delta) in ((0.5, 0.0), (1.5, 0.0), (2.5, 0.6878340181193607), (3.0, 2.9987857021037776),
                       (5.5, 2log(10) * 5.5 - C))
        gamma = sqrt(1 + 10.0^(2x)); tau = gamma - 1; beta2 = 1 - 1 / gamma^2
        energy = tau * co.electron_mc2
        bracket = log(2 * (gamma + 1) / (AIR_I / co.electron_mc2)^2) + P._F(ElectronState, tau, cl.Tcut / co.electron_mc2, beta2)
        actualdelta = bracket - P.energy_loss(cl, ElectronState, energy) / (2π * co.r_e^2 * co.electron_mc2 * AIR_NEL / beta2)
        @test actualdelta ≈ delta atol=1e-6
    end

    # The Chebyshev version reproduces the exact friction force, which acts against p.
    cl = ContinuumLoss(AIR_NEL, AIR_I, 1000co.eV)
    ccl = ChebContinuumLoss(cl, 1e9co.eV, 3)
    for ev in (1.2e3, 5e3, 1e5, 1e6, 3e7), S in (ElectronState, PositronState)
        s = state(S, S == ElectronState ? Electron : Positron, ev * co.eV)
        f1, f2 = P.force(cl, s), P.force(ccl, s)
        @test dot(f1, s.p) < 0
        @test f1 ≈ f2 rtol=2e-2
        @test norm(f1) ≈ P.energy_loss(cl, S, kinenergy(s)) rtol=1e-12
    end
    # No friction on photons
    for f in (cl, ccl)
        @test iszero(P.force(f, state(PhotonState, Photon, 1e6co.eV)))
    end
end

@testset "Collision tables" begin
    # Photo-electric absorption: the Chebyshev fit reproduces the cross section.
    for z in (7, 8)
        process = PhotoElectric(z)
        table = collision_table_from_processes([(co.nair, process)], Photon, 0.0)
        for kev in (60, 150, 200, 300)
            energy = kev * 1000co.eV
            fitted = P.rate(table, 1, P.presample(table, nothing, energy))
            # The table has one cubic per octave in energy, which is too coarse for the steep
            # E^-3 photo-electric cross section: errors up to ~12 % at 60-300 keV (irrelevant while
            # photons below the 100 keV cut are dropped).
            @test fitted ≈ co.nair * co.c * totalcs(process, energy) rtol=0.15
        end
    end

    @test isfinite(totalcs(Compton(7), 1e6co.eV))
    # KNOWN BUG: the Z <= 1 branch of Compton uses an undefined `keV` (hydrogen is never used).
    @test_broken isfinite(totalcs(Compton(1), 1e6co.eV))
    @test totalcs(Compton(7), 0.0) == 0
    # The Klein-Nishina formula agrees with the Compton parametrisation to a few percent where both
    # are valid. KleinNishinaCompton is NaN at E = 0 (it is never called below 0.1 keV).
    for mev in (0.1, 1.0, 10.0, 100.0)
        @test totalcs(KleinNishinaCompton(7), mev * 1e6co.eV) ≈ totalcs(Compton(7), mev * 1e6co.eV) rtol=0.03
    end
    @test_broken isfinite(totalcs(KleinNishinaCompton(7), 0.0))

    # Moller/Bhabha cross sections vanish below the cut
    @test all(iszero, totalcs.(Ref(Moller(7, 1000co.eV)), [0, 999, 1999] .* co.eV))
    @test all(iszero, totalcs.(Ref(Bhaba(7, 1000co.eV)), [500] .* co.eV))
    # KNOWN EDGE CASES: exactly at the cut (Moller: 1/(1-x) with x=1) and at E = 0 (Bhabha: x=Inf)
    # the cross section is NaN instead of 0. The tables never evaluate these points.
    @test_broken iszero(totalcs(Moller(7, 1000co.eV), 1000co.eV))
    @test_broken iszero(totalcs(Bhaba(7, 1000co.eV), 0.0))
    @test totalcs(Moller(7, 1000co.eV), 1e6co.eV) > 0
    @test totalcs(Bhaba(7, 1000co.eV), 1e6co.eV) > 0

    # A Chebyshev table is a smooth fit; check it away from the discontinuity at the cut.
    for (process, species) in ((Moller(7, 50e3co.eV), Electron), (Bhaba(7, 50e3co.eV), Positron))
        table = collision_table_from_processes([(co.nair, process)], species, 0.0;
                                               max_energy=1e7co.eV, safety=1.0)
        for energy in (1e3, 1e4, 1e5, 2e5, 1e6, 5e6) .* co.eV
            pre = P.presample(table, nothing, energy)
            fitted = P.rate(table, 1, pre)
            actual = co.nair * speed(species, energy) * totalcs(process, energy)
            if energy > 2 * 50e3co.eV * 1.5
                @test fitted ≈ actual rtol=2e-2
            end
            @test fitted <= P.ratebound(table, energy) * (1 + 1e-12)
        end
    end

    b = BinaryIntervals(10, 1.0)
    coeff = chebfit(x -> 1 + x, b, 3)
    @test P.chebeval(0.999, b, coeff) ≈ 1.999   # (x = xmax exactly is one past the last interval)
end

@testset "LXCAT slow-electron table" begin
    cross = open(io -> parse_bolsigplus(io), joinpath(DATA, "Cross_section_N2_O2.txt"))
    kinds = Set(c[:kind] for c in cross)
    @test kinds ⊆ Set(["EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATIONAL"])
    @test all(c -> c[:target] in ("N2", "O2"), cross)
    @test all(c -> issorted(first.(c[:data])) && all(x -> x[2] >= 0, c[:data]), cross)

    Fdt = co.elementary_charge * 3e5 * 2.5e-11
    table = collision_table_from_lxcat(cross, SlowElectron, Fdt; safety=1.15)
    @test length(table) == length(cross)
    # LogLinRange is x = exp(L) - x0 with x0 = 1 eV: the grid is [0, 109 eV], not [1, 110] eV.
    @test first(table.energy) ≈ 0.0 atol=1e-30
    @test last(table.energy) ≈ 109co.eV rtol=1e-9
    @test all(isfinite, table.rate) && all(>=(0), table.rate)
    @test all(isfinite, table.ratebound)

    # The rate bound must cover the total rate at every grid point.
    total = vec(sum(table.rate, dims=1))
    @test all(table.ratebound .>= total .* (1 - 1e-12))

    # EFFECTIVE has been converted into the elastic cross section: it can not exceed the original
    # effective one and cannot be negative.
    for (j, proc) in enumerate(table.proc)
        proc isa Elastic && @test all(>=(0), table.rate[j, :])
    end
    @test count(p -> p isa Elastic, table.proc) == count(c -> c[:kind] == "EFFECTIVE", cross)

    # presample works at both ends of the grid and interpolates between neighbouring points
    for e in (1co.eV, 5co.eV, 50co.eV, 108.9co.eV)
        k, w = P.presample(table, nothing, e)
        @test 1 <= k < length(table.energy) && 0 <= w <= 1
        @test P.ratebound(table, e) >= sum(P.rate(table, j, (k, w)) for j in 1:length(table)) * (1 - 1e-12)
    end
end

@testset "Slow-electron collision processes" begin
    Random.seed!(7)
    K = 30co.eV
    s = slowstate(K; w=2.0, t=3e-9)
    for _ in 1:200
        o = P.collide(Excitation(8co.eV), s, kinenergy(s))
        @test o isa StateChangeOutcome
        @test kinenergy(o.state) ≈ K - 8co.eV rtol=1e-12
        @test o.state.w == 2
        @test o.state.t == s.t                       # the clock must not be overwritten

        o = P.collide(Elastic(1e-4), s, kinenergy(s))
        # Elastic scattering off a heavy target changes direction but (almost) not energy.
        @test kinenergy(o.state) ≈ K rtol=1e-3
        @test o.state.t == s.t

        o = P.collide(Ionization(15.6co.eV), s, kinenergy(s))
        @test o isa NewParticleOutcome
        # Remaining energy is split equally between both electrons
        @test kinenergy(o.state1) ≈ (K - 15.6co.eV) / 2 rtol=1e-12
        @test kinenergy(o.state2) ≈ (K - 15.6co.eV) / 2 rtol=1e-12
        @test o.state1.t == s.t && o.state2.t == s.t   # incl. the newly created electron
    end
    @test P.collide(Attachment(0.0), s, kinenergy(s)) isa RemoveParticleOutcome
end
