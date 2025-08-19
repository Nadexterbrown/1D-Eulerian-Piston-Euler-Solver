import os

import numpy as np
import cantera as ct

from pyro import Pyro
from pyro.util import msg
import pyro.mesh.boundary as bnd

from utilities.general import *
from utilities.pele import *



########################################################################################################################
#
########################################################################################################################

GAMMA_FALLBACK = 1.4
P_FLOOR = 1e-12
RHO_FLOOR = 1e-12
A_FLOOR = 1e-12

########################################################################################################################
#
########################################################################################################################

def phi_from_density(x, rho, t, up0=1626.35, A=None, f=45.4e3):
    """
    x : 1D array of cell centers [m]
    rho : 1D array of density at those centers [kg/m^3]
    t : time [s]
    returns φ with φ=0 at the piston position x_p(t)
    """
    if A is None:
        A = 0.2 * up0
    dx = np.diff(x)
    rho_avg = 0.5 * (rho[:-1] + rho[1:])
    Phi_left = np.concatenate(([0.0], np.cumsum(rho_avg * dx)))  # kg/m^2

    def piston_position(t, up0=1626.35, A=None, f=45.4e3):
        if A is None:
            A = 0.2 * up0
        if t <= 0.0:
            return 0.0
        # integrate u_p(t) = up0 + A sin(2π f t) with x_p(0)=0
        # x_p(t) = up0*t + (A/(2π f))*(1 - cos(2π f t))
        return up0 * t + (A / (2.0 * np.pi * f)) * (1.0 - np.cos(2.0 * np.pi * f * t))

    xp = piston_position(t, up0=up0, A=A, f=f)

    # linear extrapolation
    slope_left = rho[0]
    slope_right = rho[-1]
    if xp <= x[0]:
        Phi_xp = Phi_left[0] + (xp - x[0]) * slope_left
    elif xp >= x[-1]:
        Phi_xp = Phi_left[-1] + (xp - x[-1]) * slope_right
    else:
        Phi_xp = np.interp(xp, x, Phi_left)

    return Phi_left - Phi_xp

########################################################################################################################
# Compressible Eulerian Piston Solver
########################################################################################################################

class pyroSolver:

    # Allowed BCs Pyro understands (adjust if your fork differs)
    _VALID_BC = {"outflow", "inflow", "periodic", "reflect", "user"}

    def __init__(self,
                 sim_time=1e-3,
                 cfl=0.8,
                 nx=100,
                 ny=1,
                 xmax=0.1,
                 ymax=0.01,
                 bc=None,
                 piston=None,
                 output_dir='output'
                 ):
        # Store the arguments
        self.sim_time = sim_time  # total simulation time in seconds
        self.cfl = cfl  # CFL number
        self.nx = nx  # number of grid points in x-direction
        self.ny = ny  # number of grid points in y-direction
        self.xmax = xmax  # maximum x-coordinate
        self.ymax = ymax  # maximum y-coordinate
        self.output_dir = output_dir  # directory to save output files

        # Initialize boundary conditions with sensible defaults and merge user input
        default_bc = {"xl": "outflow", "xr": "outflow", "yl": "outflow", "yr": "outflow"}
        if bc is None:
            # No user-specified BCs; start from the defaults
            self.bc = default_bc.copy()
        else:
            # Merge user BCs with defaults so unspecified directions fall back to "outflow"
            self.bc = default_bc.copy()
            self.bc.update(bc)

        # Initialize piston parameters
        if piston is None:
            self.piston = {
                'Kind': 'constant',  # Type of piston motion ('constant' or 'sine')
                'Initial Velocity': 1500.0,  # Initial piston velocity in m/s
                'Amplitude': 0,  # Amplitude for oscillatory motion
                'Frequency': 0,  # Frequency for oscillatory motion in Hz
                'Ramp Time': 0,  # Ramp time for smooth start in seconds
            }

    def _configure_parameters(self):
        # Step 1: Create the cantera gas object with the specified mechanism
        self.gas = ct.Solution(input_params.mech)
        self.gas.TPX = input_params.T, input_params.P, input_params.X
        self.gas()
        # Step 2: Set the parameter dictionary for the pyro solver
        self.parameters = {
            "driver.verbose": 0,
            "driver.tmax": self.sim_time,
            "driver.cfl": self.cfl,

            "compressible.limiter": 2,

            "mesh.nx": self.nx,
            "mesh.ny": self.ny,
            "mesh.xmax": self.xmax,
            "mesh.ymax": self.ymax,

            "mesh.xlboundary": self.bc["xl"] ,
            "mesh.xrboundary": self.bc["xr"],
            "mesh.ylboundary": self.bc["yl"],
            "mesh.yrboundary": self.bc["yr"],

            "ic.density": self.gas.density_mass,
            "ic.velocity": 0.0,
            "ic.pressure": self.gas.P,

            "piston.kind": self.piston['Kind'],  # Type of piston motion
            "piston.initialVelocity": self.piston['Initial Velocity'],  # Initial piston velocity in m/s
            "piston.amplitude": self.piston['Amplitude'],  # Amplitude for oscillatory motion
            "piston.frequency": self.piston['Frequency'],  # Frequency for oscillatory motion in Hz
            "piston.rampTime": self.piston['Ramp Time'],  # Frequency for oscillatory motion in Hz

            "eos.gamma": self.gas.cp_mass / self.gas.cv_mass,  # Use Cantera's cp/cv ratio as gamma
        }


    def setup(self):
        # Step 1: Ensure output directory exists
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, f'Density'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, f'Velocity'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, f'Pressure'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, f'Internal-Energy'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, f'Comparison'), exist_ok=True)

        # Step 2: Configure simulation parameters
        self._configure_parameters()

        # Step 3: Define the solver used for the compressible euler equation
        self.p = Pyro("compressible")

        # Step 4: Define solver specific boundary condition routines
        bnd.define_bc("piston", user, is_solid=False)

        # Step 5: Register the problem with the solver
        self.p.add_problem("piston", init_data, problem_params=self.parameters)

        # Step 6: Define the simulation and initial conditions
        self.p.initialize_problem("piston", inputs_dict={"driver.verbose": 1})
        print("solid.xl, solid.xr, solid.yl, solid.yr =",
              int(self.p.sim.solid.xl), int(self.p.sim.solid.xr),
              int(self.p.sim.solid.yl), int(self.p.sim.solid.yr))

        # Step 7: Setup result list
        self.result_list = []


    def run(self, time_stepping=False):
        if time_stepping:
            idx = 0
            time = float(0.0)
            while time < self.parameters["driver.tmax"]:
                # Advance the simulation one time step
                self.p.single_step()

                # Extract results
                time = self.p.sim.cc_data.t

                if idx % 100 == 0:
                    # Get the Pyro variable object (for grid info)
                    Density_var = self.p.get_var("density")
                    X_Vel_Var = self.p.get_var("velocity")[0]
                    Pressure_var = self.p.get_var("pressure")
                    Energy_var = self.p.get_var("energy")

                    # Extract 1D arrays (averaged over y)
                    Density = np.average(Density_var.v(), axis=1).copy()
                    X_Velocity = np.average(X_Vel_Var.v(), axis=1).copy()
                    Pressure = np.average(Pressure_var.v(), axis=1).copy()
                    IntEnergy = (np.average(Energy_var.v(), axis=1).copy() - 0.5 * Density * (
                                X_Velocity ** 2)) / Density

                    # Get x-coordinates from the grid
                    x_coords = Density_var.g.x[Density_var.g.ilo: Density_var.g.ihi + 1]

                    # Compute T(x,t) from density and pressure
                    # Compute temperature for each cell efficiently

                    gas = ct.Solution(input_params.mech)
                    Temperature = np.zeros_like(Density)
                    for i in range(len(Density)):
                        gas.DPX = Density[i], Pressure[i], input_params.X
                        Temperature[i] = gas.T

                    # Compute phi(x,t) from density
                    Phi = phi_from_density(x_coords, Density, time)

                    snapshot = {
                        "Time": time,
                        "X": x_coords.copy(),
                        "Phi": Phi.copy(),
                        "Density": Density.copy(),
                        "Velocity": X_Velocity.copy(),
                        "Pressure": Pressure.copy(),
                        "Temperature": Temperature.copy(),
                        "Int_Energy": IntEnergy.copy()
                    }
                    self.result_list.append(snapshot)

                    # Save plots if enabled
                    if self.output_dir:
                        plot_single(x_coords, Density, title=f"Density at t={time*1e3:.2f} ms",
                                    xlabel="Position (m)", ylabel="Density (kg/m³)",
                                    output=os.path.join(self.output_dir, f'Density/Frame-{idx}.png'))
                        plot_single(x_coords, X_Velocity, title=f"Velocity at t={time*1e3:.2f} ms",
                                    xlabel="Position (m)", ylabel="Velocity (m/s)",
                                    output=os.path.join(self.output_dir, f'Velocity/Frame-{idx}.png'))
                        plot_single(x_coords, Pressure, title=f"Pressure at t={time*1e3:.2f} ms",
                                    xlabel="Position (m)", ylabel="Pressure (Pa)",
                                    output=os.path.join(self.output_dir, f'Pressure/Frame-{idx}.png'))
                        plot_single(x_coords, IntEnergy, title=f"Internal Energy at t={time*1e3:.2f} ms",
                                    xlabel="Position (m)", ylabel="Internal Energy (J/kg)",
                                    output=os.path.join(self.output_dir, f'Internal-Energy/Frame-{idx}.png'))
                        plot_multiple(x_coords, (Density / np.max(Density),
                                                 Pressure / np.max(Pressure),
                                                 Temperature / np.max(Temperature),
                                                 IntEnergy / np.max(IntEnergy),
                                                 X_Velocity / np.max(X_Velocity)),
                                      title=f"Comparison Plot at t={time*1e3:.2f} ms",
                                      xlabel="Position (m)",
                                      ylabel=("Density", "Pressure", "Temperature", "Internal Energy", "Velocity"),
                                      output=os.path.join(self.output_dir, f'Comparison/Frame-{idx}.png'))
                idx += 1
        else:
            self.p.run_sim()

        self.p.finalize()
        msg.info("Simulation complete.")


def user(bc_name, bc_edge, variable, ccdata):
    """
        Apply custom boundary conditions.

        Parameters
        ----------
        bc_name : {'piston'}
            Descriptive name for the boundary condition.
        bc_edge : {'xlb', 'xrb', 'ylb', 'yrb'}
            Which boundary is being updated.
        variable : {'density', 'x-momentum', 'y-momentum', 'energy'}
            The variable whose ghost cells we are filling.
        ccdata : CellCenterData2d
            The data object holding simulation variables.
    """

    ############################################################
    # Internal/Helper Functions
    ############################################################

    def _get_gamma(ccdata, default=GAMMA_FALLBACK):
        """
        Retrieve γ used by the solver, without relying on ccdata.params.
        Tries, in order:
          1) attribute 'gamma'
          2) aux field 'gamma' (scalar or length-1 array)
          3) provided default
        """
        # 1) attribute
        g = getattr(ccdata, "gamma", None)
        if g is not None:
            try:
                return float(g)
            except Exception:
                pass

        # 2) aux field
        try:
            aux = ccdata.get_var("gamma")
            # scalar or length-1 array
            if np.isscalar(aux):
                return float(aux)
            arr = np.asarray(aux)
            if arr.size == 1:
                return float(arr.ravel()[0])
        except Exception:
            pass

        # 3) fallback
        return float(default)

    def _interior_prims(ccdata, i):
        """
        Return interior primitive slices at x-index i (vector over y):
          rh  : density (>= RHO_FLOOR)
          u,v : velocities
          Et  : total energy
        """
        dens = ccdata.get_var("density")
        xmom = ccdata.get_var("x-momentum")
        ymom = ccdata.get_var("y-momentum")
        ener = ccdata.get_var("energy")

        rho = dens[i, :]
        rh = np.maximum(rho, RHO_FLOOR)

        u = xmom[i, :] / rh
        v = ymom[i, :] / rh
        Et = ener[i, :]

        return rh, u, v, Et

    def _pressure_from_cons(gamma, rho, u, v, Etot):
        """
        Compute pressure from conservative variables:
           p = (γ-1) * (E_tot - 1/2 ρ (u^2+v^2))
        Applies kinetic-energy clipping to avoid negative internal energy
        and floors pressure to P_FLOOR.
        Accepts arrays or scalars.
        """
        ke = 0.5 * rho * (u * u + v * v)
        eint = np.maximum(Etot - ke, 0.0)
        p = (gamma - 1.0) * eint
        return np.maximum(p, P_FLOOR)

    def _min_resolvable_ramp(ccdata, myg, gamma, k_cells=10):
        """
        Estimate a numerically resolvable ramp time: k_cells * dx / a0.
        Uses myg.dx directly and a robust a0 average from the first few interior rows.
        """
        dx = myg.dx

        # sample a0 from the first 2 interior columns (guard against tiny/NaN)
        ilo = myg.ilo
        cols = [ilo, min(ilo + 1, myg.ihi)]
        a_samples = []
        for i in cols:
            rho_i, u_i, v_i, E_i = _interior_prims(ccdata, i)
            p_i = _pressure_from_cons(gamma, rho_i, u_i, v_i, E_i)
            a_i = np.sqrt(np.maximum(gamma * p_i / np.maximum(rho_i, RHO_FLOOR), 0.0))
            a_samples.append(a_i)

        a0 = float(np.mean(np.maximum(np.mean(a_samples, axis=0), A_FLOOR)))

        tau = k_cells * dx / max(a0, A_FLOOR)

        # keep it within a sensible range for startup (avoid absurdly large/ small)
        return float(np.clip(tau, 5e-9, 5e-6))

    def _ghost_state_char_moc_cached(ccdata, myg, piston_params):
        """
        LEFT-wall piston: build a thermodynamically consistent ghost state
        using method-of-characteristics with J^- taken from the interior.

        Caches the state at the current time so density/xmom/ymom/energy fills
        all see the exact same wall state.
        """
        tnow = float(getattr(ccdata, "t", 0.0))
        cache = getattr(ccdata, "_piston_cache", None)
        if cache is not None and cache.get("t") == tnow and cache.get("mode") == "char_moc":
            return cache["rho_g"], cache["u_g"], cache["v_g"], cache["p_g"], cache["E_g"]

        gamma = _get_gamma(ccdata, default=GAMMA_FALLBACK)
        ilo = myg.ilo

        # Interior primitives & pressure
        rho_i, u_i, v_i, E_i = _interior_prims(ccdata, ilo)
        p_i = _pressure_from_cons(gamma, rho_i, u_i, v_i, E_i)

        rh = np.maximum(rho_i, RHO_FLOOR)
        a_i = np.maximum(np.sqrt(np.maximum(gamma * p_i / rh, 0.0)), A_FLOOR)

        # Isentropic constant K = p / rho^gamma  (assume smooth wall, no entropy jump)
        K = p_i / (rh ** gamma)

        # OUTGOING (from interior to the left wall) invariant for a LEFT boundary:
        # C^- leaves the domain; simple C^+ wave -> J^- is constant. Take J^- from interior.
        Jm = u_i - 2.0 * a_i / (gamma - 1.0)

        # Minimum resolvable ramp time (for the piston law):
        pp = piston_params.copy()
        if pp.get('ramp_time') > 0:
            tau_min = _min_resolvable_ramp(ccdata, myg, gamma, k_cells=10)
            pp['ramp_time'] = max(pp.get('ramp_time', 0.0), tau_min)

        # Wall speed from your piston law (parameters are inside piston_params)
        u_b = piston_bc(tnow, **pp)

        # Solve for wall sound speed from J^-:
        #   u_b - 2 a_b / (γ-1) = Jm_int  =>  a_b = 0.5 (γ-1) (u_b - Jm_int)
        a_b = 0.5 * (gamma - 1.0) * (u_b - Jm)
        a_b = np.maximum(a_b, A_FLOOR)

        # Isentropic recovery: a_b^2 = γ p_b / ρ_b  and  p_b = K ρ_b^γ
        rho_b = ((a_b * a_b) / (gamma * np.maximum(K, P_FLOOR))) ** (1.0 / (gamma - 1.0))
        rho_b = np.maximum(rho_b, RHO_FLOOR)
        p_b = K * (rho_b ** gamma)

        v_b = v_i
        E_b = p_b / (gamma - 1.0) + 0.5 * rho_b * (u_b * u_b + v_b * v_b)

        if np.any(p_b < p_i):
            print(f"[WARN] MoC wall p_b < interior p_i at t={tnow:.3e} — indicates rarefaction.")

        # cache for consistency across per-variable fills
        ccdata._piston_cache = {
            "t": tnow, "mode": "char_moc",
            "rho_g": rho_b, "u_g": u_b, "v_g": v_b, "p_g": p_b, "E_g": E_b
        }
        return rho_b, u_b, v_b, p_b, E_b

    ############################################################
    # Main Function
    ############################################################

    """
    LEFT-wall piston via MoC. Writes ONLY ghost cells for the requested variable.
    """
    myg = ccdata.grid
    if bc_name != "piston" or bc_edge != "xlb":
        return
    if variable not in ("density", "x-momentum", "y-momentum", "energy"):
        return

    # --- Choose ONE set for your case (parameters live here, not globals) ---
    # Constant-speed piston (Fig. 4-like):
    piston_params = dict(
        kind="constant",
        U=1500.0,
        A=0.0,
        f=0.0,
        ramp_time=0.0,
    )

    # Or oscillatory (Fig. 6-like):
    # piston_params = dict(kind="sine", U0=1626.35, A=0.2*1626.35, f=45.4e3, ramp_time=1.0e-9)

    # Build/reuse a consistent ghost state once
    rho_g, u_g, v_g, p_g, E_g = _ghost_state_char_moc_cached(ccdata, myg, piston_params)

    xmom_g = rho_g * u_g
    ymom_g = rho_g * v_g

    ilo, ng = myg.ilo, myg.ng
    for i in range(ilo - 1, ilo - ng - 1, -1):
        if variable == "density":
            ccdata.get_var("density")[i, :] = rho_g
        elif variable == "x-momentum":
            ccdata.get_var("x-momentum")[i, :] = xmom_g
        elif variable == "y-momentum":
            ccdata.get_var("y-momentum")[i, :] = ymom_g
        elif variable == "energy":
            ccdata.get_var("energy")[i, :] = E_g


def init_data(my_data, rp):
    if rp.get_param("driver.verbose"):
        msg.bold("initializing the accelerating piston problem...")

    # -----------------------------
    # Far-field initial conditions
    # -----------------------------
    rho0 = float(rp.get_param("ic.density"))
    u0   = float(rp.get_param("ic.velocity"))
    p0   = float(rp.get_param("ic.pressure"))
    gamma = float(rp.get_param("eos.gamma"))

    # If provided, use a piston speed for the first interior cell (skip ramp).
    # Fallback: try "piston.U" or default to 1500 m/s.
    u_p = float(rp.params.get("piston.initialVelocity"))

    # Field handles
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # -----------------------------
    # Uniform far-field fill
    # -----------------------------
    dens[:] = rho0
    xmom[:] = rho0 * u0
    ymom[:] = 0.0
    ener[:] = p0 / (gamma - 1.0) + 0.5 * rho0 * (u0**2)

    # ----------------------------------------------------------------------
    # Set the FIRST ACTIVE INTERIOR COLUMN to the piston state (no ramp):
    #   - isentropy K = p / rho^gamma
    #   - constant J^- : u - 2 a/(γ-1) = const  (far-field has u0)
    #   => a1 = a0 + (γ-1)/2 * (u_p - u0)
    #   => rho1 from a1^2 = γ p1 / rho1 and p1 = K rho1^γ
    # ----------------------------------------------------------------------
    g    = my_data.grid
    ilo  = g.ilo + 10
    iref = min(ilo + 1, g.ihi)  # reference cell for robust far-field (in case ilo is modified later)

    # use the specified far-field (rho0,p0) for the compatibility; alternatively,
    # we could read them back from arrays at iref.
    K    = p0 / (max(rho0, RHO_FLOOR)**gamma)
    a0   = np.sqrt(max(gamma * p0 / max(rho0, RHO_FLOOR), 0.0))
    a0   = max(a0, A_FLOOR)

    # simple-wave relation from far field to piston cell
    a1   = a0 + 0.5 * (gamma - 1.0) * (u_p - u0)
    a1   = max(a1, A_FLOOR)

    rho1 = ((a1*a1) / (gamma * max(K, P_FLOOR))) ** (1.0 / (gamma - 1.0))
    rho1 = max(rho1, RHO_FLOOR)
    p1   = K * (rho1 ** gamma)

    # write the first interior column (vector in y)
    dens[:ilo, :] = rho1
    xmom[:ilo, :] = rho1 * u_p
    ymom[:ilo, :] = 0.0
    ener[:ilo, :] = p1 / (gamma - 1.0) + 0.5 * rho1 * (u_p*u_p)


def piston_bc(t: float,
              kind: str = "constant",
              U: float = 1500.0,
              A: float = 0.0,
              f: float = 0.0,
              ramp_time: float = 1e-8) -> float:
    """
    Time-dependent piston velocity u_p(t) with an internal C^2 ramp.

    Parameters
    ----------
    t : float
        Current time [s].
    kind : {"constant","sine"}
        Type of motion.
    U : float
        Constant velocity [m/s] for kind="constant".
    A : float
        Amplitude [m/s] for kind="sine".
    f : float
        Frequency [Hz] for kind="sine".
    ramp_time : float
        Ramp-in duration [s] for smooth start (0 disables ramp).
    U0 : float
        Mean velocity [m/s] for kind="sine".

    Returns
    -------
    float
        Wall speed u_p(t) [m/s].
    """

    def ramp_smooth(t, ramp_time):
        """
        Sigmoid/tanh ramp from 0 to 1 over [0, ramp_time].
        Centered at 0.5*ramp_time for symmetric rise.
        """
        if ramp_time <= 0.0:
            return 1.0
        arg = 6.0 * (t - 0.5 * ramp_time) / ramp_time  # 95% rise within ramp_time
        return 0.5 * (1.0 + np.tanh(arg))

    if kind == "sine":
        vl = U + A * np.sin(2.0 * np.pi * f * t)
    else:  # constant
        vl = U

    if ramp_time == 0.0:
        return vl
    else:
        return vl * ramp_smooth(t, ramp_time)

########################################################################################################################
# Main Function
########################################################################################################################

def main():
    # Step 1: Set the gas properties and create cantera gas object
    initialize_parameters(
        T=300,
        P=ct.one_atm,
        Phi=1.0,
        Fuel='H2',
        nitrogenAmount=0.5 * 3.76,
        mech='../Li-Dryer-H2-mechanism.yaml',
    )  # ✅ shared state

    # Step 3: Initialize the pyroSolver with the gas object and output directory
    simulation = pyroSolver(
        sim_time=8.5e-5,  # total simulation time in seconds
        cfl=0.1,  # CFL number
        nx=1024,  # number of grid points in x-direction
        ny=8,  # number of grid points in y-direction
        xmax=0.01,  # maximum x-coordinate
        ymax=0.005,  # maximum y-coordinate
        bc={"xl": "piston", "xr": "outflow", "yl": "periodic", "yr": "periodic"},  # boundary conditions
        output_dir='Oscillating-Piston-Data-Animation-Frames'  # directory to save output files
    )

    simulation.setup()
    simulation.run(time_stepping=True)


    return

if __name__ == "__main__":
    main()