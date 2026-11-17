using LinearAlgebra
 
"""
Lorentz force q (E + v × B) on electrons and positrons. `efield` and `bfield` are functions
`(x, t) -> SVector{3}` in V/m and tesla. Photons feel no force: the fallback
`force(::AbstractForcing, s)` in pusher.jl returns zero for any particle type not defined here.
"""
struct FieldForcing{E, B} <: Particulator.AbstractForcing
    efield::E
    bfield::B
end
 
# Returns dp/dt in N, the same quantity ContinuumLoss.force returns.
function Particulator.force(f::FieldForcing, s::ElectronState)
    v = Particulator.velocity(s)
    return -co.elementary_charge * (f.efield(s.x, s.t) + cross(v, f.bfield(s.x, s.t)))
end
 
function Particulator.force(f::FieldForcing, s::PositronState)
    v = Particulator.velocity(s)
    return  co.elementary_charge * (f.efield(s.x, s.t) + cross(v, f.bfield(s.x, s.t)))
end
 
 
"""
Build the pusher. Uniform field `Ez` (V/m) between `z1` and `z2` (zero outside), as in the old
DoubleLayerField, plus a magnetic field (zero by default) and optional continuous losses.
"""
function build_pusher(; Ez, z1, z2, B = SA[0.0, 0.0, 0.0], composition, Tcut, I = 85.7 * co.eV,
                      continuous_loss = true)
    Evec = SA[0.0, 0.0, -Ez]            # same sign convention as the old script
    zero3 = SA[0.0, 0.0, 0.0]
 
    efield = (x, t) -> (z1 <= x[3] <= z2) ? Evec : zero3
    bfield = (x, t) -> B
 
    lorentz = FieldForcing(efield, bfield)
    if !continuous_loss
        return RK2Pusher(lorentz)
    end
 
    # Electron density: 14 electrons per N2, 16 per O2.
    nel = 14 * composition["N2"] + 16 * composition["O2"]
    friction = ContinuumLoss(Float64(nel), Float64(I), Float64(Tcut))
    return RK2Pusher(CombinedForcing(lorentz, friction))
end
