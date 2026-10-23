# Particulator.jl

[![Build Status](https://github.com/aluque/Particulator.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/aluque/Particulator.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Particulator.jl** is a 3D relativistic Monte Carlo particle transport code in Julia for simulating coupled populations of **electrons ($e^-$)**, **positrons ($e^+$)**, and **photons ($\gamma$, X-rays and gamma rays)** propagating and multiplying in a gaseous medium under electric and magnetic fields.

It is built primarily for high-energy atmospheric and plasma physics, including:
* **Relativistic Runaway Electron Avalanches (RREAs)** in thunderstorm electric fields
* **Terrestrial Gamma-ray Flashes (TGFs)** and **thunderstorm gamma-ray glows**
* **Relativistic feedback** driven by backward-propagating positrons and bremsstrahlung gamma rays
* **High-energy electron, positron, and photon beams** in air and other gas mixtures

## Physics Overview

`Particulator.jl` tracks individual super-particles in 3D space and relativistic momentum space $(\mathbf{x}, \mathbf{p}, t)$ with statistical weights $w$, from sub-keV cutoff energies up to hundreds of MeV. Between discrete collisions, particles evolve under external electromagnetic fields and continuous stopping power; at collision events, secondary electrons, positrons, and photons are generated self-consistently across populations.

### 1. Relativistic Dynamics and Continuous Energy Loss

Charged particles ($e^-$ and $e^+$) with rest mass $m_e$ and charge $q = \pm e$ obey the relativistic equations of motion:

$$\frac{d\mathbf{x}}{dt} = \mathbf{v} = \frac{\mathbf{p}}{\gamma m_e}, \qquad \frac{d\mathbf{p}}{dt} = q\left(\mathbf{E}(\mathbf{x}, t) + \mathbf{v} \times \mathbf{B}(\mathbf{x}, t)\right) + \mathbf{F}_{\text{loss}}(\mathbf{p})$$

where $\gamma = \sqrt{1 + |\mathbf{p}|^2 / (m_e c)^2}$ is the Lorentz factor and $K = (\gamma - 1)m_e c^2$ is the kinetic energy. Photons propagate ballistically at speed $c$ with velocity $\mathbf{v} = c\,\mathbf{p}/|\mathbf{p}|$ and energy $E_\gamma = c|\mathbf{p}|$.

When impact ionization is modeled discretely only above a kinetic energy threshold $T_{\text{cut}}$ (as in Møller or Bhabha scattering), soft energy losses from excitations and sub-threshold ionizations ($W < T_{\text{cut}}$) are included as a continuous friction force $\mathbf{F}_{\text{loss}} = -(dE/dx)\,\mathbf{p}/|\mathbf{p}|$. `ContinuumLoss` (and its Chebyshev-interpolated counterpart `ChebContinuumLoss`) implements the **Bethe-Bloch restricted stopping power** for electrons and positrons following the Geant4 Physics Reference Manual (Chapter 11):

$$-\frac{dE}{dx} = \frac{2\pi r_e^2 m_e c^2 n_{\text{el}}}{\beta^2} \left[ \ln\left(\frac{2(\gamma + 1)}{(I / m_e c^2)^2}\right) + F^\pm(\tau, \tau_{\text{up}}) - \delta \right]$$

where $n_{\text{el}}$ is the medium's bound electron density, $I$ is the mean excitation energy (e.g., $I_{\text{N}_2} = 82\text{ eV}$, $I_{\text{O}_2} = 95\text{ eV}$ from ICRU Report 37), $\tau = K / (m_e c^2)$, $\tau_{\text{up}} = \min(T_{\text{cut}}/m_e c^2, \tau_{\max})$, $F^\pm$ is the Møller ($e^-$) or Bhabha ($e^+$) restricted shell term, and $\delta$ is Sternheimer's density-effect correction.

### 2. Electron Interactions ($e^-$)

* **Relativistic Screened Coulomb Scattering (`RelativisticCoulomb(Z)`):**
  Elastic angular deflection of electrons by atomic nuclei of atomic number $Z$. Uses the relativistic screened Rutherford cross section with the Wentzel-Moliere screening radius $a = 1.3413\,Z^{-1/3} a_0$ and screening parameter $\alpha = \hbar^2 / (4|\mathbf{p}|^2 a^2)$:

  $$\frac{d\sigma}{d\Omega} \propto \frac{1 - \beta^2 \sin^2(\theta/2)}{\left(\sin^2(\theta/2) + \alpha\right)^2}$$

* **Relativistic Binary-Encounter-Bethe Impact Ionization (`RBEB(B, U, N)` and `ORBITALS`):**
  Discrete impact ionization of molecular and atomic orbitals down to the orbital binding energy $B$, parameterized by binding energy $B$, mean orbital kinetic energy $U$, and electron occupation number $N$.
  * Pre-configured orbital sets for molecular nitrogen (`ORBITALS["N2"]`) and molecular oxygen (`ORBITALS["O2"]`) combine valence molecular orbitals from Hwang et al., *J. Chem. Phys.* **104**, 2956 (1996) with K-shell ($1s$) orbitals from Santos et al., *J. Phys. B* **36**, 4211 (2003).
  * Because `RBEB` resolves ionization all the way down to the valence ionization threshold (~12--15 eV), no artificial continuous stopping power is needed for electrons when `RBEB` is used.
  * Secondary and scattered electron polar angles follow relativistic two-body kinematics (Lehtinen et al., 1999):

    $$\cos\theta_{1,2} = \sqrt{\frac{E_{1,2}(E_0 + 2m_e c^2)}{E_0(E_{1,2} + 2m_e c^2)}}$$

* **Møller Scattering (`Moller(Z, tcut)`):**
  Relativistic free electron-electron impact ionization producing knock-on electrons with kinetic energy $E_2 \ge T_{\text{cut}}$ (Geant4 Physics Reference Manual, Chapter 10). Intended to be used alongside `ContinuumLoss` for energy transfers below $T_{\text{cut}}$.

* **Bremsstrahlung Emission (`SeltzerBerger(Z)`):**
  Radiative emission of a photon ($e^- + Z \to e^- + \gamma + Z$) in the electrostatic field of target nuclei ($Z = 1\text{--}100$).
  * Differential and total cross sections are interpolated from the **Seltzer & Berger (1986)** scaled bremsstrahlung tables (*Atom. Data Nucl. Data Tables* **35**, 345--418), included in `data/brem_SB/` for elements $Z = 1$ to $100$.
  * Emitted photon polar angles are sampled from Geant4's modified Tsai distribution (`G4ModifiedTsai`), and the primary electron recoils with momentum $\mathbf{p}_{e}' = \mathbf{p}_e - \mathbf{p}_\gamma$.

### 3. Positron Interactions ($e^+$)

* **Relativistic Screened Coulomb Scattering (`RelativisticCoulomb(Z)`):**
  Elastic nuclear scattering using the same screened relativistic Coulomb formulation as electrons.
* **Bhabha Impact Ionization (`Bhaba(Z, tcut)`):**
  Relativistic positron-electron scattering ($e^+ + e^- \to e^+ + e^-$) liberating secondary electrons with kinetic energy above $T_{\text{cut}}$ (Geant4 Physics Reference Manual, Chapter 10). Typically paired with `ContinuumLoss` (restricted to `PositronState`) to account for soft energy losses below $T_{\text{cut}}$.
* **In-Flight and At-Rest Two-Photon Annihilation (`PositronAnihilation(Z)`):**
  * **In flight:** Annihilation of a relativistic positron with an atomic electron ($e^+ + e^- \to \gamma_1 + \gamma_2$) using the Heitler two-quantum annihilation cross section and exact relativistic energy-angle kinematics (Geant4 Physics Reference Manual, Section 10.3).
  * **At rest (sub-cutoff):** Whenever a positron's kinetic energy drops below the population cutoff energy $K_{\text{thresh}}$, `drop(::PositronState)` automatically annihilates it at rest into two counter-propagating $511\text{ keV}$ ($m_e c^2$) photons emitted isotropically.

### 4. Photon Interactions ($\gamma$)

* **Photoelectric Absorption (`PhotoElectric(Z)`):**
  Absorption of a photon by an atom ($Z = 1\text{--}100$) accompanied by the emission of a photoelectron ($E_e = E_\gamma - B_{\text{shell}}$).
  * Total cross sections are computed from the analytical piecewise polynomial fits of Biggs & Lighthill (1990) / Sandia National Laboratories (`G4StaticSandiaData`), with atomic shell binding energies from `src/atomic_shells.jl`.
  * Photoelectron emission angles follow the relativistic **Sauter-Gavrila distribution** as formulated in the PENELOPE manual.
* **Compton (Incoherent) Scattering (`Compton(Z)` and `KleinNishinaCompton(Z)`):**
  Inelastic photon scattering off atomic electrons ($\gamma + e^- \to \gamma' + e^-$), generating a scattered photon and a recoil electron.
  * `Compton(Z)` uses Geant4's empirical parameterization (`G4KleinNishinaCompton`), which matches the Klein-Nishina cross section at high energies ($E_\gamma \gtrsim 100\text{ keV}$) while accounting for atomic binding suppression at low photon energies.
  * `KleinNishinaCompton(Z)` uses the exact bare-electron Klein-Nishina total cross section (Landau & Lifshitz Vol. 4, Section 86).
* **Bethe-Heitler Electron-Positron Pair Production (`BetheHeitler(Z)`):**
  Conversion of a high-energy photon ($E_\gamma > 2m_e c^2 \approx 1.022\text{ MeV}$) into an $e^- + e^+$ pair in the field of a nucleus and its atomic electrons ($\sigma \propto Z(Z + 1)$).
  * Cross sections, atomic screening functions, and high-energy Coulomb corrections ($E_\gamma > 50\text{ MeV}$) follow Geant4's `G4BetheHeitlerModel`, with lepton emission angles sampled from the modified Tsai distribution.

### 5. Summary of Physical Processes

| Species | Process Type | Constructor | Physical Model & Reference | Target Density Scaling |
| :-- | :-- | :-- | :-- | :-- |
| $e^-$ | Elastic scattering | `RelativisticCoulomb(Z)` | Screened Rutherford/Mott (Wentzel-Moliere) | Per atom (`2 * n_N2`) |
| $e^-$ | Bremsstrahlung | `SeltzerBerger(Z)` | Seltzer & Berger (1986) tables + Tsai angle | Per atom (`2 * n_N2`) |
| $e^-$ | Impact ionization | `RBEB(B, U, N)` / `ORBITALS` | Relativistic Binary-Encounter-Bethe (Hwang 1996; Santos 2003) | Per molecule (`n_N2`) |
| $e^-$ | Impact ionization | `Moller(Z, tcut)` | Relativistic Møller scattering ($E_2 \ge T_{\text{cut}}$) | Per atom (`2 * n_N2`) |
| $e^+$ | Elastic scattering | `RelativisticCoulomb(Z)` | Screened Rutherford/Mott (Wentzel-Moliere) | Per atom (`2 * n_N2`) |
| $e^+$ | Impact ionization | `Bhaba(Z, tcut)` | Relativistic Bhabha scattering ($E_2 \ge T_{\text{cut}}$) | Per atom (`2 * n_N2`) |
| $e^+$ | Annihilation | `PositronAnihilation(Z)` | Heitler 2$\gamma$ in-flight + automatic 2$\times$511 keV at rest | Per atom (`2 * n_N2`) |
| $\gamma$ | Photoelectric effect | `PhotoElectric(Z)` | Sandia/Biggs-Lighthill CS + Sauter-Gavrila angle | Per atom (`2 * n_N2`) |
| $\gamma$ | Compton scattering | `Compton(Z)` / `KleinNishinaCompton(Z)` | Klein-Nishina with/without low-$E$ binding correction | Per atom (`2 * n_N2`) |
| $\gamma$ | Pair production | `BetheHeitler(Z)` | Screened Bethe-Heitler ($E_\gamma > 2m_e c^2$) | Per atom (`2 * n_N2`) |
| $e^-, e^+$ | Continuous stopping | `ContinuumLoss(nel, I, Tcut)` | Restricted Bethe-Bloch ($W < T_{\text{cut}}$) + density effect | Electron density `nel` |

> **Important Note on Target Densities (Atomic vs. Molecular):**
> Processes parameterized by atomic number `Z` (`RelativisticCoulomb(7)`, `SeltzerBerger(7)`, `PhotoElectric(7)`, `Compton(7)`, `BetheHeitler(7)`, `Moller(7, tcut)`, `Bhaba(7, tcut)`, `PositronAnihilation(7)`) return cross sections **per atom**. For diatomic molecules such as $\text{N}_2$ ($Z=7$) and $\text{O}_2$ ($Z=8$), multiply the molecular number density by **2** when building collision tables: `(2 * comp["N2"], RelativisticCoulomb(7))`.
> Conversely, the `RBEB` orbitals in `ORBITALS["N2"]` and `ORBITALS["O2"]` already account for all electrons in the molecule, so their cross sections are **per molecule**: `(comp["N2"], orb)`.

### 6. Electromagnetic Field Configurations

* `HomogeneousField(v::SVector{3, T})`: Spatially and temporally uniform field $\mathbf{F}(\mathbf{x}, t) = \mathbf{v}$.
* `DoubleLayerField(z1, z2, v::SVector{3, T})`: Uniform field $\mathbf{v}$ confined to the altitude slab $z_1 < z < z_2$, and zero outside.
* `StepField(z0, v1::SVector{3, T}, v2::SVector{3, T})`: Field equal to $\mathbf{v}_1$ for $z < z_0$ and $\mathbf{v}_2$ for $z \ge z_0$.
* `ConfinedDoubleLayerField(sx, sy, sz, ez0)`: 3D analytical divergence-free approximation to a finite thunderstorm double layer with Gaussian widths $s_x, s_y, s_z$ and peak axial field $E_{z0}$:

  $$E_x = -\frac{E_{z0}\, x z}{s_x^2} e^{-\left(\frac{x^2}{2s_x^2} + \frac{y^2}{2s_y^2} + \frac{z^2}{2s_z^2}\right)}, \quad E_y = -\frac{E_{z0}\, y z}{s_y^2} e^{-\left(\frac{x^2}{2s_x^2} + \frac{y^2}{2s_y^2} + \frac{z^2}{2s_z^2}\right)}, \quad E_z = E_{z0}\left(1 - \frac{z^2}{s_z^2}\right) e^{-\left(\frac{x^2}{2s_x^2} + \frac{y^2}{2s_y^2} + \frac{z^2}{2s_z^2}\right)}$$

* **Custom Fields:** Any Julia function or callable struct `f(x, t) -> SVector{3, Float64}` can be passed to `ElectromagneticField(efield, bfield)`.

### 7. Units and Physical Constants

All inputs and outputs in `Particulator.jl` are in **SI units**:
* **Length / Position (`x`):** meters ($\text{m}$)
* **Time (`t`, `dt`):** seconds ($\text{s}$)
* **Energy (`K`):** Joules ($\text{J}$)
* **Momentum (`p`):** $\text{kg}\cdot\text{m}/\text{s}$
* **Electric Field (`E`):** Volts per meter ($\text{V}/\text{m}$)
* **Magnetic Field (`B`):** Tesla ($\text{T}$)
* **Number Density (`n`):** $\text{m}^{-3}$

The module `Particulator.Constants` (aliased as `Particulator.co`) provides physical constants and unit conversions:
* `co.eV`: $1.6021766\times 10^{-19}\text{ J}$ (multiply eV by `co.eV` to convert to Joules)
* `co.nair`: $2.6867811\times 10^{25}\text{ m}^{-3}$ (Loschmidt constant / standard air molecular density at STP)
* `co.c`: $2.99792458\times 10^8\text{ m/s}$ (speed of light)
* `co.electron_mass`, `co.elementary_charge`, `co.electron_mc2`, `co.r_e`, `co.a_0`, `co.fine_structure`

## Getting Started: Simulating an RREA in Air

### Installation

Requires **Julia 1.11** or later. Clone the repository and instantiate the dependencies:

```bash
git clone https://git.app.uib.no/David.Sarria/particulator.jl
cd particulator.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Launch Julia with multi-threading enabled:

```bash
julia --threads=auto --project=.
```

### Complete Example: Relativistic Runaway Electron Avalanche

The script below simulates a $1\text{ MeV}$ seed electron injected into standard air ($79\%\text{ N}_2$, $21\%\text{ O}_2$) inside a $350\text{ kV/m}$ thunderstorm electric field layer between $z = 0\text{ m}$ and $z = 2000\text{ m}$:

```julia
using Particulator
using StaticArrays
using Random

const co = Particulator.co

function run_rrea()
    Random.seed!(0)

    # 1. Physical parameters (SI units)
    dt          = 2.5e-11              # Integration timestep: 25 ps
    tfinal      = 1.0e-8               # Simulation duration: 10 ns
    efield_mag  = 3.5e5                # Ambient electric field: 350 kV/m
    Kthresh     = 1e3 * co.eV          # Low-energy tracking cutoff: 1 keV
    init_energy = 1e6 * co.eV          # Seed electron kinetic energy: 1 MeV
    maxp        = 1_000_000            # Max super-particles per species
    safety      = 1.15                 # Rate-bound safety factor

    # Dry air molecular densities at STP
    comp = Dict(
        "N2" => co.nair * 0.79,
        "O2" => co.nair * 0.21
    )

    # Max momentum impulse gained from the electric field in one timestep dt
    Fdt = co.elementary_charge * efield_mag * dt

    # 2. Build collision tables for electrons, positrons, and photons
    e_processes = [
        (2 * comp["N2"], RelativisticCoulomb(7)),
        (2 * comp["O2"], RelativisticCoulomb(8)),
        (2 * comp["N2"], SeltzerBerger(7)),
        (2 * comp["O2"], SeltzerBerger(8)),
        [(comp["N2"], orb) for orb in ORBITALS["N2"]]...,
        [(comp["O2"], orb) for orb in ORBITALS["O2"]]...
    ]
    ecolls = collision_table_from_processes(e_processes, Electron, Fdt; safety)

    p_processes = [
        (2 * comp["N2"], RelativisticCoulomb(7)),
        (2 * comp["O2"], RelativisticCoulomb(8)),
        (2 * comp["N2"], Bhaba(7, Kthresh)),
        (2 * comp["O2"], Bhaba(8, Kthresh)),
        (2 * comp["N2"], PositronAnihilation(7)),
        (2 * comp["O2"], PositronAnihilation(8))
    ]
    pcolls = collision_table_from_processes(p_processes, Positron, Fdt; safety)

    photon_processes = [
        (2 * comp["N2"], PhotoElectric(7)),
        (2 * comp["O2"], PhotoElectric(8)),
        (2 * comp["N2"], BetheHeitler(7)),
        (2 * comp["O2"], BetheHeitler(8)),
        (2 * comp["N2"], Compton(7)),
        (2 * comp["O2"], Compton(8))
    ]
    gcolls = collision_table_from_processes(photon_processes, Photon, 0.0)

    # 3. Initialize seed electron and species populations
    # Note: include a tiny transverse momentum component (1e-6 * pnorm) when
    # launching along the z-axis to avoid a coordinate singularity at sin(theta) = 0.
    pnorm = Particulator.momentum_norm_from_kin(Electron, init_energy)
    p0    = SA[0.0, 1e-6 * pnorm, pnorm]
    x0    = SA[0.0, 0.0, 0.0]

    electrons = Population(maxp, [ElectronState(x0, p0)], ecolls, Kthresh)
    photons   = Population(maxp, PhotonState{Float64}[], gcolls, Kthresh)
    positrons = Population(maxp, PositronState{Float64}[], pcolls, Kthresh)

    mpopl = MultiPopulation(
        :electron => electrons,
        :photon   => photons,
        :positron => positrons
    )
    init!(mpopl)

    # 4. Set up electromagnetic fields and continuous stopping power for positrons
    # E along -z accelerates negatively charged electrons toward +z
    efield  = DoubleLayerField(0.0, 2000.0, SA[0.0, 0.0, -efield_mag])
    bfield  = HomogeneousField(SA[0.0, 0.0, 0.0])
    emfield = ElectromagneticField(efield, bfield)

    # Mean excitation energy I (ICRU Report 37) and bound electron density nel
    I_air = (comp["N2"] * 82 + comp["O2"] * 95) * co.eV / (comp["N2"] + comp["O2"])
    n_el  = comp["N2"] * 14 + comp["O2"] * 16
    closs = ChebContinuumLoss(ContinuumLoss(n_el, I_air, Kthresh), 1e9 * co.eV, 3)
    ploss = RestrictedForcing{PositronState}(closs)

    pusher = RK2Pusher(CombinedForcing(emfield, ploss))

    # 5. Configure diagnostics and Russian roulette population control
    counts_cb   = ParticleCountCallback([:electron, :photon, :positron])
    roulette_cb = RouletteCallback(10_000)
    counter_cb  = CollisionCounter()
    callback    = CombinedCallback((counts_cb, roulette_cb, counter_cb))

    # 6. Run the simulation
    run!(mpopl, pusher, tfinal, dt, callback; output_dt = 1e-9, verbosity = 1)

    return mpopl, counts_cb, counter_cb
end

mpopl, counts_cb, counter_cb = run_rrea()
```

### Extracting Physical Observables

After `run!` completes (or inside a callback during the run), you can inspect the physical state of any population:

```julia
electrons = get(mpopl, Electron)

# Total physical number of electrons (sum of statistical weights w)
total_physical_electrons = Particulator.weight(electrons)

# Number of active Monte Carlo super-particles in memory
n_superparticles = nactives(electrons)

# Weighted mean kinetic energy (converted from Joules to MeV)
mean_energy_MeV = Particulator.meanenergy(electrons) / (1e6 * co.eV)

# Spatial center-of-mass (centroid vector) and RMS spread (in meters)
centroid_m, rms_spread_m = Particulator.spread(electrons)

# Position variance along (x, y, z)
var_xyz = Particulator.posvar(electrons)

# Extract kinetic energies (keV), positions (m), and weights of active particles
energies_keV = [kinenergy(p) / (1e3 * co.eV) for p in eachactive(electrons)]
z_coords_m   = [p.x[3] for p in eachactive(electrons)]
weights      = [p.w for p in eachactive(electrons)]
```

To record particles crossing a detector plane (for example, $z = 500\text{ m}$), attach a `WallCallback`:

```julia
# Axis 3 is z; drop=false keeps particles alive after crossing the plane
z_detector = WallCallback{PhotonState{Float64}}(3, 500.0, false)
```

Whenever a photon crosses $z = 500\text{ m}$, `WallCallback` linearly interpolates the particle's position, momentum, weight, and time to the exact plane crossing and appends the state to `z_detector.accum`.

## Numerical Methods and Code Architecture

### 1. Null-Collision Method with Dynamic Rate Bounds
In an external electric field, a charged particle's kinetic energy $K$ changes continuously between collisions, which means its total collision rate $\nu_{\text{tot}}(K) = \sum_j n_j v(K) \sigma_j(K)$ also varies along its trajectory.

`Particulator.jl` handles this exactly using the **null-collision (thinning) method**:
1. Each particle carries a dimensionless optical depth to its next collision, $s \sim \text{Exp}(1)$, and a local upper-bound collision rate $r$ computed at the start of the timestep $\Delta t$.
2. Because a particle with charge $q$ in an electric field $E$ can gain at most momentum $\Delta p = |q| E \Delta t \equiv F_{\Delta t}$ during a timestep, its energy can change by at most $\Delta K \approx \gamma^3 v F_{\Delta t}$. `collision_table_from_processes` precomputes a rate upper bound (`compratebound`):

   $$r(K) = \nu_{\text{tot}}(K) + s_{\text{safety}}\, \gamma^3 v(K) F_{\Delta t} \left|\frac{d\nu_{\text{tot}}}{dK}\right|$$

3. During a timestep $\Delta t$, the time to the next candidate collision is $\Delta t_{\text{coll}} = s / r$. The particle is pushed forward by $\min(\Delta t_{\text{rem}}, \Delta t_{\text{coll}})$ using Ralston's 2nd-order Runge-Kutta integrator (`RK2Pusher`).
4. If a collision occurs within the step, a random number $\xi \in [0, r)$ selects either a physical process $j$ (with probability $\nu_j(K_{\text{new}}) / r$) or a `NullCollision` (with probability $1 - \nu_{\text{tot}}(K_{\text{new}}) / r$). Collision processes are sorted from highest to lowest maximum rate and unrolled at compile time via a `@generated` function (`do_one_collision!`) for branch efficiency.

### 2. Fast $O(1)$ Cross-Section Lookups via Binary-Interval Chebyshev Expansions
Cross sections vary rapidly at low energies (near ionization thresholds) and smoothly at high energies. Instead of using large linear grids or slow binary searches (`searchsorted`), `ChebyshevCollisionTable` partitions the energy domain $(0, E_{\max}]$ into $k$ logarithmically spaced binary intervals (`BinaryIntervals`):

$$I_0 = \left(0,\, 2^{-k} E_{\max}\right], \qquad I_i = \left(2^{i - k - 1} E_{\max},\, 2^{i - k} E_{\max}\right] \quad (i = 1, \dots, k)$$

Within each binary interval, the collision rate $\nu_j(K)$ is approximated by a low-order Chebyshev polynomial expansion (`chebfit`, default order 3). At runtime, `precheb` extracts the floating-point binary exponent of $K / E_{\max}$ in $O(1)$ hardware instructions using `frexp`, mapping $K$ directly to its interval index $i$ and local coordinate $\xi \in [-1, 1]$ without any conditional search loop.

### 3. Statistical Weight Control (Russian Roulette and Splitting)
In exponentially growing avalanches, simulating every secondary particle with unit weight quickly exhausts memory. Conversely, in sub-threshold or strongly attenuating regimes, particle counts may dwindle. `Particulator.jl` assigns each super-particle a statistical weight `w` and provides two unbiased variance-reduction techniques:
* **Russian Roulette (`roulette!(p_survive, popl)` and `RouletteCallback(target_n)`):** When the active super-particle count $N_{\text{act}}$ exceeds `target_n`, each particle is retained with probability $p = N_{\text{target}} / N_{\text{act}}$ (and its weight is scaled to $w / p$) or deactivated with probability $1 - p$.
* **Particle Splitting (`split!(mean_new, popl)` and `SplitCallback{S}(target_n, min_frac)`):** When the active super-particle count of species `S` falls below `target_n * min_frac`, each active super-particle is split into $1 + \text{Poisson}(\lambda)$ copies with reduced weight $w / (1 + \lambda)$.

### 4. Memory Layout, Multi-Threading, and Callbacks
* **Structure of Arrays (`StructArrays.jl`):** Each `Population` stores its particles in a `StructVector{PS}` so fields like `active`, `x`, `p`, `w`, `t`, `s`, and `r` are stored in contiguous arrays. Inactive slots created by absorbed or dropped particles are compacted in-place via `repack!(popl)`.
* **Low-Overhead Threading (`Polyester.jl`):** Particle advancement and collisions are parallelized across CPU threads with `@batch`, while secondary particles created during a timestep are appended thread-safely using atomic index increments (`Atomic{Int}`) and advanced in subsequent sub-sweeps until all secondaries reach `t + dt`.
* **Callbacks (`AbstractCallback`):** Custom diagnostics, boundary conditions, or live visualizations can be implemented by subtyping `AbstractCallback` and overloading `oncollision`, `onadvance`, `onstep`, or `onoutput` (see `dev/withplot.jl` for an interactive `GLMakie` live-plotting callback).

## Repository Structure

* `src/`: Core package source files:
  * `electron.jl`, `positron.jl`, `photon.jl`, `particledefs.jl`: Relativistic kinematics and state definitions.
  * `relativistic_coulomb.jl`, `rbeb.jl`, `moller.jl`, `seltzer.jl`: Electron scattering, ionization, and bremsstrahlung models.
  * `bhaba.jl`, `anihilation.jl`: Positron ionization and annihilation models.
  * `photo_electric.jl`, `static_sandia_data.jl`, `atomic_shells.jl`, `compton.jl`, `bethe_heitler.jl`: Photon absorption, scattering, and pair production models.
  * `continuum.jl`: Bethe-Bloch restricted stopping power (`ContinuumLoss`, `ChebContinuumLoss`).
  * `field.jl`, `pusher.jl`: Electromagnetic field geometries, forcings, and `RK2Pusher`.
  * `cheby.jl`, `collision_table.jl`, `collisions.jl`: Binary-interval Chebyshev interpolation and null-collision sampling.
  * `population.jl`, `mixed_population.jl`, `callback.jl`, `run.jl`: Population management, Russian roulette/splitting, callbacks, and main simulation loop.
  * `slow-electron.jl`, `zhelezniak_photon.jl`, `lxcat.jl`: Non-relativistic electron transport, Zhelezniak photoionization photons, and LXCat JSON cross-section loader.
* `data/`:
  * `data/brem_SB/`: Seltzer-Berger bremsstrahlung cross-section tables (`br1` to `br100`).
  * `data/xcom/`: NIST XCOM photon cross-section data for $Z = 1, 7, 8$.
* `dev/`: Current example scripts (`dev/swarm.jl` for RREA swarm simulations and `dev/withplot.jl` for live `GLMakie` visualization).
* `scripts/`: Earlier standalone scripts (`scripts/beam.jl`, `scripts/swarm.jl`). Note: these two scripts predate `RK2Pusher` and use an older `run!(mpopl, efield, bfield, ...)` signature; use `dev/swarm.jl` as the reference for the current API.

## License

MIT License. Copyright (c) 2025 Alejandro Luque Estepa (`aluque@iaa.es`) and contributors. See [`LICENSE`](LICENSE) for details.
