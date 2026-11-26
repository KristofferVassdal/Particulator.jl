##
##  DATA INPUT AND PRE-PROCESSING
##
"""
Reads a collision database from a .json file and builds a collision
table with cumulative cross sections scaled by the densities contained
in the densities dict and evaluated at the grid eng. 

We use two parameters to control photo-ionization noise:

    * `photon_weight` is the weight of photons. Reducing this reduces the photo-ionization noise,
      beyond what is physically expected.
    * `photon_multiplier` is a factor that increases the probability of photo-emission
      and correspondingly reduces the number of photons emitted (when photon_multiplier = 1
      the expected number of emitted photons is the electron weight.      
"""

"""
This module contains the code required to parse BOLSIG+-compatible files.

Direct translation of the BOLSIG+ parser from the bolos package by Alejandro Luque. 


Most users would only call `parse_bolsigplus`, documented below.
"""

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

# BOLSIG+'s user guide says that separators must consist of at least five dashes
const RE_SEP = r"-----+"

# Matches an optional '<' followed by '->', i.e. '->' or '<->'
const RE_ARROW = r"<?->"

"""
    _loadtxt(lines)

Parses whitespace-delimited numeric rows from a vector of lines, mirroring
`np.loadtxt(lines).tolist()`: returns a vector of rows (each a
`Vector{Float64}`).
"""
function _loadtxt(lines::Vector{String})
    isempty(lines) && return Vector{Float64}[]
    text = join(lines, "\n")
    matrix = readdlm(IOBuffer(text))
    return [collect(Float64, row) for row in eachrow(matrix)]
end

"""
    _read_until_sep(fp)

Reads lines from `fp` until a separator line is found.
"""
function _read_until_sep(fp)
    lines = String[]
    while !eof(fp)
        line = readline(fp)
        if occursin(RE_SEP, strip(line))
            break
        end
        push!(lines, strip(line))
    end
    return lines
end

"""
    _read_block(fp; has_arg=true)

Reads the data of a process, contained in a block. `has_arg` indicates
whether an argument line has to be read.
"""
function _read_block(fp; has_arg::Bool=true)
    target = strip(readline(fp))
    arg = has_arg ? strip(readline(fp)) : nothing

    comment = join(_read_until_sep(fp), "\n")

    @debug "Read process '$target'"
    data = _loadtxt(_read_until_sep(fp))

    return target, arg, comment, data
end

# ---------------------------------------------------------------------------
# Specialized functions for each keyword. They all return dictionaries with
# the relevant attributes.
# ---------------------------------------------------------------------------

"""
    _read_momentum(fp)

Reads a MOMENTUM, ELASTIC or EFFECTIVE block.
"""
function _read_momentum(fp)
    target, arg, comment, data = _read_block(fp; has_arg=true)
    mass_ratio = Base.parse(Float64, split(arg)[1])

    d = Dict(:target => target,
             :mass_ratio => mass_ratio,
             :comment => comment,
             :data => data)

    return d
end

"""
    _read_excitation(fp)

Reads an EXCITATION or IONIZATION block.
"""
function _read_excitation(fp)
    target, arg, comment, data = _read_block(fp; has_arg=true)

    parts = [strip(s) for s in split(target, RE_ARROW)]
    lhs, rhs = parts[1], parts[2]

    d = Dict(:target => lhs,
             :product => rhs,
             :comment => comment,
             :data => data)

    if "<->" in split(target)
        argvals = split(arg)
        threshold = Base.parse(Float64, argvals[1])
        weight_ratio = Base.parse(Float64, argvals[2])
        d[:weight_ratio] = weight_ratio
    else
        threshold = Base.parse(Float64, split(arg)[1])
    end

    d[:threshold] = threshold
    return d
end

"""
    _read_attachment(fp)

Reads an ATTACHMENT block.
"""
function _read_attachment(fp)
    target, _, comment, data = _read_block(fp; has_arg=false)

    d = Dict(:comment => comment,
             :data => data,
             :threshold => 0.0)

    lr = [strip(s) for s in split(target, RE_ARROW)]

    if length(lr) == 2
        d[:target] = lr[1]
        d[:product] = lr[2]
    else
        d[:target] = target
    end

    return d
end

# ---------------------------------------------------------------------------
# Keyword dispatch table — must come after the _read_* functions above,
# since Julia needs them to already exist to build this Dict.
# ---------------------------------------------------------------------------

const KEYWORDS = Dict(
    "MOMENTUM"   => _read_momentum,
    "ELASTIC"    => _read_momentum,
    "EFFECTIVE"  => _read_momentum,
    "EXCITATION" => _read_excitation,
    "IONIZATION" => _read_excitation,
    "ATTACHMENT" => _read_attachment,
    "THREE" => _read_attachment,
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

"""
    parse_bolsigplus(fp)

Parses a BOLSIG+ cross-sections file.

# Arguments
- `fp`: an IO object (e.g. an opened file) positioned at the start of the
  file.

# Returns
- `processes::Vector{Dict}`: a list of dictionaries, one per process, as
  found in the file.
"""
function parse_bolsigplus(fp)
    processes = Dict[]
    while !eof(fp)
        line = readline(fp)
        key = strip(line)
        if haskey(KEYWORDS, key)
            fread = KEYWORDS[key]
            @debug "New process of type '$key'"

            d = fread(fp)
            d[:kind] = key
            push!(processes, d)
        end
    end

    @info "Parsing complete. $(length(processes)) processes read."

    return processes
end



function load_lxcat(fnames, densities, energy;
                    photon_weight=1.0,
                    photon_multiplier=1.0)
    T = eltype(energy)
    
    db = mapreduce(vcat, fnames) do fname
        local db1    
        open(fname, "r") do fd
            db1 = JSON.parse(read(fd, String))
        end
        return db1
    end
    
    # We first build a dictionary with the targets to later correct for the
    # EFFECTIVE / ELASTIC problem
    targets = Dict()

    v = @. sqrt(2 * energy / mass(Electron()))

    nprocs = 0
    for item in db
        dens::Float64 = get(densities, item["target"], 0.0)
        dens == 0.0 && continue
        target = get!(targets, item["target"], Vector())

        energy0 = [v[1] * co.eV for v in item["data"]]
        cs0 = [v[2] for v in item["data"]]

        if occursin("3-body", item["comment"])
            cs0 .*= densities[item["target"]] ./ co.centi^-3
        end

        if haskey(item, "rescale")
            cs0 .*= item["rescale"]
        end

        if haskey(item, "weight_scale")
            cs0 .*= item["weight_scale"]
        end

        if item["kind"] == "PHOTOEMISSION"
            @info "PHOTOEMISSION detected.  Scaling cross section by " photon_multiplier
            cs0 .*= photon_multiplier
        end
        
        # For performance first we interpolate to an uniform grid
        itp = extrapolate(interpolate((energy0,), cs0, Gridded(Linear())),
                          Flat())
        
        item["nu"] = @. dens * v * itp.(energy)

        push!(target, item)
        nprocs += 1
    end

    proc = Vector{CollisionProcess}()
    
    dkinds = Dict("ATTACHMENT" =>
                  (itm, dens) -> Attachment(itm["threshold"] * co.eV),

                  "EXCITATION" =>
                  (itm, dens) -> Excitation(itm["threshold"] * co.eV),

                  "IONIZATION" =>
                  (itm, dens) -> Ionization(itm["threshold"] * co.eV),

                  "ELASTIC" =>
                  (itm, dens) -> Elastic(itm["mass_ratio"]),

                  "PHOTOEMISSION" => (itm, dens) -> begin
                  # The absorption is rescaled according to the density in the
                  # "absorber" field and transformed from 1/m to 1/s.
                  νmin = co.c * dens[itm["absorber"]] * itm["chi_min"]
                  νmax = co.c * dens[itm["absorber"]] * itm["chi_max"]

                  PhotoEmission(νmin, νmax, photon_multiplier, photon_weight)
                  end

                  )


    νrun = zeros(T, length(energy))
    rate = zeros(T, (nprocs + 1, length(energy)))
    i = 1

    for (name, ps) in targets
        # Ensure that we have elastic collisions and not effective
        ensure_elastic(ps)
        
        for item in ps
            push!(proc, dkinds[item["kind"]](item, densities))
            rate[i, :] .= item["nu"]
            @info "New process: " * signature(item)
            i += 1
        end
    end

    νtotal = dropdims(sum(rate, dims=1), dims=1)
    
    # Find also the max. of the collision rate
    maxrate = maximum(νtotal)

    nullrate = maxrate .- νtotal
    rate[nprocs + 1, :] = nullrate
    push!(proc, NullCollision())
    
    # Order collisions according to the integral (sum) over energies
    inteng = dropdims(sum(rate, dims=2), dims=2)
    perm = sortperm(inteng, rev=true)
    origperm = invperm(perm)
    
    proc = tuple(proc[perm]...)
    rate = rate[perm, :]
    
    @assert size(rate, 1) == length(proc)
    
    (;proc, rate, maxrate, origperm)
end

"""
Prepare a photo-emission process.
"""
function prepare_photo_emission(itm, dens)
end


""" 
Make sure that we have elastic collisions in the collision set for a target.
"""
function ensure_elastic(procs)
    for p in procs
        p["kind"] == "EFFECTIVE" || continue
        for p2 in procs
            if p != p2 && (p2["kind"] == "EXCITATION" || p2["kind"] == "IONIZATION")
                p["nu"] .-= p2["nu"]
            end
        end
        p["kind"] = "ELASTIC"
    end
end

    
function signature(proc)
    target = proc["target"]
    product = haskey(proc, "product") ? proc["product"] : ""
    kind = proc["kind"]
    "e + $target -> $product... ($kind)"
end
