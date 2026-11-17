# swarm.jl : Tue Feb 18 22:59:01 2025
"""
# Swarm
Investigating swarms under an electric field.

## Running the code

```julia
julia> includet("swarm.jl")     # Needs Revise.jl
julia> Swarm.main()
```
"""
module Swarm


using Particulator
const co = Particulator.co
using StaticArrays
using StatsBase
using Accessors

using Random
using Polyester
using Base.Threads: Atomic
using PyPlot
using JLD2
using Dates
using Printf



function main(;n_init_particles=2000,
              maxp=80_000_000,
              ntarget=30000,
              ntarget_slow=30000,
              init_energy=7e6 * co.eV,
              dt=2.5e-11,
              efield = 4.3e5,
              safety = 1.15,
              z1=0.0,
              z2=1e9,
              killthresh = 1e3 * co.eV,
              seed = 0,
              Kthresh = 1e5 * co.eV,
              zwall=nothing,
              nloops = 22,
              tstep = 1e-7,
              save_every = 5,
              outdir = joinpath(@__DIR__, "..", "runs",
                                "swarm_" * Dates.format(now(), "yyyymmdd_HHMMSS")),
              composition = Dict("N2" => co.nair * 0.79,
                                 "O2" => co.nair * 0.21)
              )
    Polyester.reset_threads!()
    Random.seed!(seed)
    mc2 = co.electron_mass * co.c^2
    Fdt = co.elementary_charge * efield * dt
    Lxcat_path = joinpath(@__DIR__, "..", "data", "Cross_section_N2_O2.txt")
    Lxcat_file = open(Lxcat_path, "r")
    cross = parse_bolsigplus(Lxcat_file)
    close(Lxcat_file)
    # Build collision tables
    lowcolls = collision_table_from_lxcat(cross, SlowElectron, Fdt; safety=safety)
    ecolls = build_electron_collision_table(composition, Fdt; safety)    
    pcolls = build_positron_collision_table(composition, Kthresh, Fdt; safety)
    γcolls = build_photon_collision_table(composition, Kthresh)
    
    # Initialize particles
    K = init_energy
    pz = sqrt(K^2 + 2K * mc2) / co.c
    p0 = SA[0.0, 1e-6 * pz, pz]  
    x = SA[0.0, 0.0, 0.0]
    init_p = [ElectronState(x, p0)]
    #init_particles = map(1:n_init_particles) do _
    #    x = SA[0.0, 0.0, 0.0]
    #    ElectronState(x, p0)
    #end
    
    # Initialize particles with a log-uniform distribution
    #λ, Emin, Emax = 7.3e6 * co.eV, 0.1e6 * co.eV, 50e6 * co.eV
    #sampleK() = Emin - λ * log(1 - rand() * (1 - exp(-(Emax - Emin) / λ)))
    #init_particles = map(1:n_init_particles) do _
    #     K  = sampleK()
    #     pz = sqrt(K^2 + 2K * mc2) / co.c
    #     ElectronState(SA[0.0, 0.0, 0.0], SA[0.0, 1e-6 * pz, pz])   # keep the small transverse part
    #end

    K_slow = 10 * co.eV
    v_slow = co.c * sqrt(1 - (mc2 / (mc2 + K_slow))^2)    # ≈ 1.88e6 m/s
    slow1  = [SlowElectronState(SA[0.0, 0.0, 0.0], SA[0.0, 1e-6 * v_slow, v_slow]) for _ in 1:100]

    Zero_cut = 300 * co.eV
    # Construct populations
    slow_electrons = Population(maxp, SlowElectronState{Float64}[], lowcolls, 1*co.eV, 100 * co.eV)
    electrons = Population(maxp, init_p, ecolls, Zero_cut)
    photons   = Population(maxp, PhotonState{Float64}[], γcolls, Kthresh)
    positrons = Population(maxp, PositronState{Float64}[], pcolls, Kthresh)

    population_index = Pair{Symbol, Any}[:electron => electrons,
                                         :slow_electron => slow_electrons,
                                         :photon => photons,
                                         :positron => positrons]
    mpopl = MultiPopulation(population_index...)
    init!(mpopl)

    I = ((composition["N2"] * 82 + composition["O2"] * 95) * co.eV /
        (composition["N2"] + composition["O2"]))
    efield_value = efield
    efield = DoubleLayerField(z1, z2, SA[0.0, 0.0, -efield])
    bfield = HomogeneousField(SA[0.0, 0.0, 0.0])
    nel = composition["N2"] * 14 + composition["O2"] * 16
    loss1 = ContinuumLoss(nel, I, Kthresh)
    loss = ChebContinuumLoss(loss1, 1e9 * co.eV, 3)

    loss = RestrictedForcing{PositronState}(loss)
    force = ElectromagneticField(efield, bfield)
    pusher = RK2Pusher(CombinedForcing(force, loss))
    
    counter = CollisionCounter()
    if !isnothing(zwall)
        callback1 = WallCallback{ElectronState{Float64}}(3, zwall)
        callback2 = WallCallback{PhotonState{Float64}}(3, zwall)
        callback3 = WallCallback{PositronState{Float64}}(3, zwall)

        callback = CombinedCallback((callback1, callback2, callback3))
    else
        callback = VoidCallback();
    end

    mkpath(outdir)
    @info "Writing output to $(abspath(outdir))"
    params = (; init_energy, efield = efield_value, dt, tstep, nloops, ntarget, ntarget_slow,
              safety, Kthresh, seed, maxp, composition = Dict(composition))

    t = 0.0
    save_snapshot(outdir, 0, t, population_index; params)
    for i in 1:nloops
        elapsed = @elapsed run!(mpopl, pusher, t + tstep, dt, callback; output_dt = nothing)
        t += tstep
        # Counts are logged *before* roulette, so growth in work per loop shows up here.
        log_loop(outdir, i, t, elapsed, population_index)
        n = nactives(electrons)
        n > ntarget && roulette!(ntarget / n, electrons)
        n_s = nactives(slow_electrons)
        n_s > ntarget_slow && roulette!(ntarget_slow / n_s, slow_electrons)
        if i % save_every == 0 || i == nloops
            save_snapshot(outdir, i, t, population_index; params)
        end
    end
    
    return NamedTuple(Base.@locals)
end


"""
    save_snapshot(outdir, loop, t, population_index; params = (;))

Write all active particles of every population to `outdir/snapshot_loopNNN.jld2`, one file
per call. Each population is a group with columns `x` (n×3, m), `p` (n×3, momentum in kg m/s;
for slow electrons this is the velocity `v` in m/s), `w` (weight), `t` (s) and `K_eV`
(kinetic energy). Root attributes: `loop`, `t`, and the run parameters in `params`.

Read back in Julia with `load(file)`, or in Python with `h5py`.
"""
function save_snapshot(outdir, loop, t, population_index; params = (;))
    fname = joinpath(outdir, @sprintf("snapshot_loop%03d.jld2", loop))
    jldopen(fname, "w") do f
        f["loop"] = loop
        f["t"] = t
        for (k, v) in pairs(params)
            f["params/$k"] = v
        end
        for (name, popl) in population_index
            act = collect(Particulator.eachactive(popl))
            n = length(act)
            vec_field = :p in fieldnames(eltype(popl.particles)) ? :p : :v
            X = Matrix{Float64}(undef, n, 3)
            P = Matrix{Float64}(undef, n, 3)
            for (j, l) in enumerate(act)
                X[j, :] .= l.x
                P[j, :] .= getproperty(l, vec_field)
            end
            g = "$name"
            f["$g/x"] = X
            f["$g/$vec_field"] = P
            f["$g/w"] = Float64[l.w for l in act]
            f["$g/t"] = Float64[l.t for l in act]
            f["$g/K_eV"] = Float64[Particulator.kinenergy(l) / co.eV for l in act]
        end
    end
    @info "Saved $fname"
    return fname
end


"""
Append one line per loop to `outdir/loop_log.csv` with the wall time of the loop and the
number of active particles in each population (before roulette).
"""
function log_loop(outdir, loop, t, elapsed, population_index)
    fname = joinpath(outdir, "loop_log.csv")
    names = first.(population_index)
    newfile = !isfile(fname)
    open(fname, "a") do io
        newfile && println(io, join(["loop", "t_s", "wall_s", ("n_" .* string.(names))...], ","))
        counts = [nactives(p) for p in last.(population_index)]
        println(io, join(Any[loop, t, round(elapsed; digits = 1), counts...], ","))
    end
    @info @sprintf("loop %d  t = %.2e s  wall = %.0f s  ", loop, t, elapsed) *
          join(["$k=$(nactives(p))" for (k, p) in population_index], "  ")
end


function build_electron_collision_table(comp, args...; kw...)
    # Tuples with density of scatterers, cross section
    
    processes = [
        (2 * comp["N2"], RelativisticCoulomb(7)),
        (2 * comp["O2"], RelativisticCoulomb(8)),
        (2 * comp["N2"], SeltzerBerger(7)),
        (2 * comp["O2"], SeltzerBerger(8)),
        [(comp["N2"], orb) for orb in ORBITALS["N2"]]...,
        [(comp["O2"], orb) for orb in ORBITALS["O2"]]...]
                 
    return collision_table_from_processes(processes, Electron, args...; kw...)
end

function build_positron_collision_table(comp, tcut, args...; kw...)
    # Tuples with density of scatterers, cross section
    processes = [(2 * comp["N2"], RelativisticCoulomb(7)),
                 (2 * comp["O2"], RelativisticCoulomb(8)),
                 (2 * comp["N2"], Bhaba(7, tcut)),
                 (2 * comp["O2"], Bhaba(8, tcut)),
                 (2 * comp["N2"], PositronAnihilation(7)),
                 (2 * comp["O2"], PositronAnihilation(8))]
                 
    return collision_table_from_processes(processes, Positron, args...; kw...)
end

function build_photon_collision_table(comp, args...; kw...)
    # Tuples with density of scatterers, cross section
    processes = [(2 * comp["N2"], PhotoElectric(7)),
                 (2 * comp["O2"], PhotoElectric(8)),
                 (2 * comp["N2"], BetheHeitler(7)),
                 (2 * comp["O2"], BetheHeitler(8)),
                 (2 * comp["N2"], Compton(7)),
                 (2 * comp["O2"], Compton(8))]
                     
    return collision_table_from_processes(processes, Photon, 0; kw...)
end


function spectrum(popl; bins = 120, Emin = 1, Emax = 1e8, n = 1,
                  zrange = (-Inf, Inf), norm_above = nothing)
    act = filter(l -> zrange[1] <= l.x[3] <= zrange[2],
                 collect(Particulator.eachactive(popl)))
    E = [Particulator.kinenergy(l) / (co.eV) for l in act]
    w = [l.w for l in act]
 
    edges = 10 .^ range(log10(Emin), log10(Emax), length = bins + 1)
    h = fit(Histogram, E, weights(w), edges)
    dE = diff(edges)
    f = h.weights ./ dE
 
    if isnothing(norm_above)
        f = f ./ n
    else
        m = edges[1:end-1] .>= norm_above * (1 - 1e-9)
        f = f ./ sum(f[m] .* dE[m])
    end
    return sqrt.(edges[1:end-1] .* edges[2:end]), f, edges
end
 
 
"""
Plot the spectrum of the electrons in `popl` (a `Population`, e.g. `res.electrons`), with the
RREA reference exp(-ε/7.3 MeV) when the curve is normalised above 1 MeV.
"""
function plotspec(popl, slow = nothing; bins = 120, norm_above = 1e6, kw...)
    # raw spectrum of the normal electrons (no normalisation yet)
    E, f, edges = spectrum(popl; bins, kw...)
    dE = diff(edges)

    # one normalisation constant, taken from the normal electrons only
    scale = 1.0
    if !isnothing(norm_above)
        m = edges[1:end-1] .>= norm_above * (1 - 1e-9)
        scale = sum(f[m] .* dE[m])
        scale > 0 || error("no normal electrons above norm_above = $norm_above eV")
    end
    f = f ./ scale

    plt.figure(figsize = (9, 6))
    ok = f .> 0
    plt.loglog(E[ok], f[ok], lw = 2.5, label = "electrons")

    fs = nothing
    if !isnothing(slow)
        _, fs, _ = spectrum(slow; bins, kw...)   # same bins, raw counts
        fs = fs ./ scale
        oks = fs .> 0
        plt.loglog(E[oks], fs[oks], lw = 2.5, label = "slow electrons")
    end

    if !isnothing(norm_above)
        λ = 7.3e6                                # keV
        sel = E .>= norm_above
        ref = exp.(-E[sel] ./ λ)
    #    plt.loglog(E[sel], ref ./ sum(ref .* dE[sel]), "--k", label = "exp(-ε/7.3 MeV)")
    end

    plt.xlabel("Energy (eV)", fontsize = 20)
    plt.ylabel("Particles / eV (normalised)", fontsize = 20)
    plt.tick_params(axis = "both", which = "major", labelsize = 16, length = 6, width = 1.2)
    plt.tick_params(axis = "both", which = "minor", length = 3)
    plt.grid(true, which = "major", alpha = 0.35)
    plt.grid(true, which = "minor", alpha = 0.12)
    plt.legend(fontsize = 18, frameon = false)
    plt.tight_layout()
    return (; E, f, fs)
end


"""
    list_snapshots(dir)

Return `(loops, files)` for the `snapshot_loopNNN.jld2` files in `dir`, sorted by loop.
"""
function list_snapshots(dir)
    found = Tuple{Int, String}[]
    for fn in readdir(dir)
        m = match(r"^snapshot_loop(\d+)\.jld2$", fn)
        isnothing(m) || push!(found, (parse(Int, m[1]), joinpath(dir, fn)))
    end
    sort!(found)
    return first.(found), last.(found)
end


"""
    snapshot_spectra(dir; species, loops, bins, Emin, Emax, norm_above)

Weighted energy spectra (particles / eV) for each snapshot in `dir` and each population in
`species`. All species in a snapshot share one normalisation: the integral of the `electron`
spectrum above `norm_above` eV (as in `plotspec`). Use `norm_above = nothing` to keep the raw
weighted counts. Snapshots without electrons above `norm_above` are skipped with a warning.

Returns `(; E, edges, spectra)` where `spectra` is a vector of
`(; loop, t, f = Dict(species => vector))`.
"""
function snapshot_spectra(dir; species = ("electron", "slow_electron"), loops = nothing,
                          bins = 120, Emin = 1.0, Emax = 1e8, norm_above = 1e6)
    edges = 10 .^ range(log10(Emin), log10(Emax), length = bins + 1)
    dE = diff(edges)
    E = sqrt.(edges[1:end-1] .* edges[2:end])
    hist(K, w) = fit(Histogram, K, weights(w), edges).weights ./ dE

    allloops, files = list_snapshots(dir)
    isempty(files) && error("no snapshot_loopNNN.jld2 files in $dir")
    spectra = []
    for (loop, file) in zip(allloops, files)
        !isnothing(loops) && !(loop in loops) && continue
        jldopen(file, "r") do d
            f = Dict(s => hist(d["$s/K_eV"], d["$s/w"]) for s in species)
            scale = 1.0
            if !isnothing(norm_above)
                fe = hist(d["electron/K_eV"], d["electron/w"])
                m = edges[1:end-1] .>= norm_above * (1 - 1e-9)
                scale = sum(fe[m] .* dE[m])
                if !(scale > 0)
                    @warn "loop $loop: no electrons above $norm_above eV, skipped"
                    return
                end
            end
            for s in species
                f[s] ./= scale
            end
            push!(spectra, (; loop, t = d["t"], f))
        end
    end
    return (; E, edges, spectra)
end


"""
    plot_snapshots(dir; species, loops, bins, Emin, Emax, norm_above)

Plot the energy spectra stored in the snapshot files in `dir` (a run folder written by
`main`), one curve per snapshot coloured by time, with a different line style per species.
Keyword arguments are passed to `snapshot_spectra`. Returns its result.

```julia
julia> Swarm.plot_snapshots("runs/swarm_20261117_101500")
julia> Swarm.plot_snapshots(dir; loops = [10, 15, 20, 22], species = ("electron",))
```
"""
function plot_snapshots(dir; species = ("electron", "slow_electron"), kw...)
    res = snapshot_spectra(dir; species, kw...)
    isempty(res.spectra) && error("nothing to plot")
    styles = Dict("electron" => "-", "slow_electron" => "--",
                  "photon" => ":", "positron" => "-.")
    cmap = plt.get_cmap("viridis")
    tmax = maximum(s.t for s in res.spectra)

    plt.figure(figsize = (9, 6))
    for s in res.spectra
        color = cmap(tmax > 0 ? s.t / tmax : 0.0)
        for (k, sp) in enumerate(species)
            f = s.f[sp]
            ok = f .> 0
            any(ok) || continue
            label = k == 1 ? @sprintf("t = %.1f µs (loop %d)", s.t * 1e6, s.loop) : nothing
            plt.loglog(res.E[ok], f[ok], get(styles, sp, "-"); color, lw = 2, label)
        end
    end
    # Line-style key for the species
    for sp in species
        plt.plot([], [], get(styles, sp, "-"); color = "gray", label = replace(sp, "_" => " "))
    end

    plt.xlabel("Energy (eV)", fontsize = 20)
    norm_above = get(kw, :norm_above, 1e6)
    plt.ylabel(isnothing(norm_above) ? "Weighted particles / eV" :
               "Particles / eV (normalised)", fontsize = 20)
    plt.tick_params(axis = "both", which = "major", labelsize = 16, length = 6, width = 1.2)
    plt.tick_params(axis = "both", which = "minor", length = 3)
    plt.grid(true, which = "major", alpha = 0.35)
    plt.grid(true, which = "minor", alpha = 0.12)
    plt.legend(fontsize = 12, frameon = false)
    plt.tight_layout()
    return res
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Swarm.main()
end

