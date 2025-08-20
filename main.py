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
        default_piston = {
                'Kind': 'constant',  # Type of piston motion ('constant' or 'sine')
                'Initial Velocity': 1500.0,  # Initial piston velocity in m/s
                'Amplitude': 0,  # Amplitude for oscillatory motion
                'Frequency': 0,  # Frequency for oscillatory motion in Hz
                'Ramp Time': 0,  # Ramp time for smooth start in seconds
                'Initial Smoothing': 0,  # 0: no smoothing, 1: tanh smoothing
                'Initial Bin Size': 0,  # Number of cells over which to apply initial smoothing
            }
        if piston is None:
            # No user-specified BCs; start from the defaults
            self.piston = default_piston.copy()
        else:
            # Merge user BCs with defaults so unspecified directions fall back to "outflow"
            self.piston = default_piston.copy()
            self.piston.update(piston)

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
            "compressible.use_flattening": 0,
            #"compressible.delta": 0.1,
            #"compressible.z0": 0.05,
            #"compressible.z1": 0.75,
            #"compressible.cvisc": 0.3,  # artificial viscosity coefficient

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

            "piston.initialSmoothing": self.piston['Initial Smoothing'],
            "piston.initialBinSize": self.piston['Initial Bin Size'],

            "piston.kind": self.piston['Kind'],  # Type of piston motion
            "piston.initialVelocity": self.piston['Initial Velocity'],  # Initial piston velocity in m/s
            "piston.amplitude": self.piston['Amplitude'],  # Amplitude for oscillatory motion
            "piston.frequency": self.piston['Frequency'],  # Frequency for oscillatory motion in Hz
            "piston.rampTime": self.piston['Ramp Time'],  # Frequency for oscillatory motion in Hz

            "eos.gamma": self.gas.cp_mass / self.gas.cv_mass,  # Use Cantera's cp/cv ratio as gamma
            "eos.cp": self.gas.cp_mass,  # Specific heat at constant pressure
            "eos.cv": self.gas.cv_mass,  # Specific heat at constant volume
        }


    def setup(self, piston_mode="solid"):
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

        # Step 3: Define the solver used for the compressible Euler equation
        self.p = Pyro("compressible")

        # Step 4: Register piston BC depending on mode
        if piston_mode == "solid":
            bnd.define_bc("piston", user_solid, is_solid=True)
        elif piston_mode == "inflow":
            bnd.define_bc("piston", user_inflow, is_solid=False)
        else:
            raise ValueError(f"Unknown piston_mode: {piston_mode}")

        # Step 5: Register the problem with the solver
        self.p.add_problem("piston", init_data, problem_params=self.parameters)

        # Step 6: Initialize the simulation
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

                if idx % 10 == 0:
                    # Get the Pyro variable object (for grid info)
                    Density_var = self.p.get_var("density")
                    X_Vel_Var = self.p.get_var("velocity")[0]
                    Pressure_var = self.p.get_var("pressure")
                    Energy_var = self.p.get_var("energy")

                    # Extract 1D arrays (averaged over y)
                    middle_idx = len(Density_var.v()[0, :]) // 2
                    Density = Density_var.v()[:, middle_idx].copy()
                    X_Velocity = X_Vel_Var.v()[:, middle_idx].copy()
                    Pressure = Pressure_var.v()[:, middle_idx].copy()
                    IntEnergy = (Energy_var.v()[:, middle_idx].copy() - 0.5 * Density * (
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
                        # Plotting each variable
                        plot_single(x_coords, Density, title=f"Density at t={time*1e3} ms",
                                    xlabel="Position (m)", ylabel="Density (kg/m³)",
                                    output=os.path.join(self.output_dir, f'Density/Frame-{idx}.png'))
                        plot_single(x_coords, X_Velocity, title=f"Velocity at t={time*1e3} ms",
                                    xlabel="Position (m)", ylabel="Velocity (m/s)",
                                    output=os.path.join(self.output_dir, f'Velocity/Frame-{idx}.png'))
                        plot_single(x_coords, Pressure, title=f"Pressure at t={time*1e3} ms",
                                    xlabel="Position (m)", ylabel="Pressure (Pa)",
                                    output=os.path.join(self.output_dir, f'Pressure/Frame-{idx}.png'))
                        plot_single(x_coords, IntEnergy, title=f"Internal Energy at t={time*1e3} ms",
                                    xlabel="Position (m)", ylabel="Internal Energy (J/kg)",
                                    output=os.path.join(self.output_dir, f'Internal-Energy/Frame-{idx}.png'))

                        # Plot comparison of all variables
                        # Compute gradient of velocity
                        grad_u = np.gradient(X_Velocity, x_coords)
                        imax = np.argmax(np.abs(grad_u))  # index of maximum gradient
                        # Define a window around the point of max gradient
                        window = 0.001 * np.max(x_coords)  # adjust as needed (meters)
                        x_center = x_coords[imax]
                        x_min, x_max = x_center - window, x_center + window

                        plot_multiple(x_coords, (Density / np.max(Density),
                                                 Pressure / np.max(Pressure),
                                                 Temperature / np.max(Temperature),
                                                 IntEnergy / np.max(IntEnergy),
                                                 X_Velocity / np.max(X_Velocity)),
                                      title=f"Comparison Plot at t={time*1e3} ms",
                                      xlabel="Position (m)",
                                      ylabel="Scaled Variables",
                                      labels=("Density", "Pressure", "Temperature", "Internal Energy", "Velocity"),
                                      x_limits=(x_min, x_max),
                                      output=os.path.join(self.output_dir, f'Comparison/Frame-{idx}.png'))
                idx += 1
        else:
            self.p.run_sim()

        self.p.finalize()
        msg.info("Simulation complete.")


def user_solid(bc_name, bc_edge, variable, ccdata):
    """
    LEFT boundary, moving solid wall at x=0:
      - u_b from piston_bc(t)
      - reflect u about u_b
      - p_g from exact wall-Riemann p_star (gamma-law)
      - fill ghost layers for the requested variable only
    """
    if bc_name != "piston" or bc_edge != "xlb":
        return
    if variable not in ("density", "x-momentum", "y-momentum", "energy"):
        return

    myg = ccdata.grid
    ilo, ng = myg.ilo, myg.ng

    # --- pull gamma (fallback if not provided via params) ---
    prm    = getattr(ccdata, "params", {})
    gamma  = float(prm.get("eos.gamma", GAMMA_FALLBACK))

    # --- interior conservative -> primitives at i = ilo ---
    rho_i  = np.maximum(ccdata.get_var("density")[ilo, :], RHO_FLOOR)
    u_i    = ccdata.get_var("x-momentum")[ilo, :] / rho_i
    v_i    = ccdata.get_var("y-momentum")[ilo, :] / rho_i
    Et_i   = ccdata.get_var("energy")[ilo, :]
    p_i    = (gamma - 1.0) * np.maximum(Et_i - 0.5 * rho_i * (u_i*u_i + v_i*v_i), 0.0)
    p_i    = np.maximum(p_i, P_FLOOR)

    # --- wall speed (lab frame) ---
    tnow = float(getattr(ccdata, "t", 0.0))
    u_b  = piston_bc(
        tnow,
        kind      = prm.get("piston.kind", "constant"),
        U         = prm.get("piston.initialVelocity", 1500.0),
        A         = prm.get("piston.amplitude", 0.0),
        f         = prm.get("piston.frequency", 0.0),
        ramp_time = prm.get("piston.rampTime", 0.0),
    )

    # --- exact wall-star pressure (1D gamma-law, solved per j) ---
    def _wall_star_pressure_exact(gamma, rho, p, u_rel, max_iter=12, tol=1e-8):
        a0   = np.sqrt(np.maximum(gamma * p / np.maximum(rho, RHO_FLOOR), 0.0))
        p_s  = np.array(p, copy=True)
        A_sh = 2.0 / ((gamma + 1.0) * np.maximum(rho, RHO_FLOOR))
        B_sh = (gamma - 1.0) / (gamma + 1.0) * p
        powR = (gamma - 1.0) / (2.0 * gamma)
        for j in range(p.size):
            ur = float(u_rel[j]); pj = float(p[j]); aj = float(a0[j]); rj = float(rho[j])
            if ur <= 0.0:
                # compression -> shock branch
                ps = max(pj - rj*aj*ur, pj)  # start at >= pj
                for _ in range(max_iter):
                    sq = np.sqrt(1.0 + 3.0*(ps - pj) / max((A_sh[j]*ps + B_sh[j]), 1e-30))
                    f  = (ps - pj) * sq / np.sqrt(max(A_sh[j]*ps + B_sh[j], 1e-30)) + ur
                    # crude df/dp (secant-like): derivative of above w.r.t ps
                    denom = max((A_sh[j]*ps + B_sh[j]), 1e-30)
                    dsdps = (1.5/np.sqrt(1.0 + 3.0*(ps - pj)/denom)) * (A_sh[j]*denom - (A_sh[j]*ps + B_sh[j])*A_sh[j])/(denom*denom)
                    df = sq/np.sqrt(denom) + (ps - pj)*dsdps
                    df = np.sign(df)*max(abs(df), 1e-12)
                    ps_new = ps - f/df
                    ps_new = max(ps_new, pj)
                    if abs(ps_new - ps) <= tol * max(ps, 1.0):
                        break
                    ps = ps_new
                p_s[j] = max(ps, P_FLOOR)
            else:
                # expansion -> rarefaction branch
                ps = max(pj - rj*aj*ur, 1e-3*pj)
                for _ in range(max_iter):
                    ratio = max(ps/pj, 1e-14)
                    f  = (2.0*aj/(gamma-1.0))*(ratio**powR - 1.0) + ur
                    df = (2.0*aj/(gamma-1.0))*(powR/max(ps, P_FLOOR))*ratio**(powR - 1.0)
                    ps_new = ps - f/max(df, 1e-12)
                    ps_new = max(ps_new, P_FLOOR)
                    if abs(ps_new - ps) <= tol * max(ps, 1.0):
                        break
                    ps = ps_new
                p_s[j] = max(min(ps, pj), P_FLOOR)  # monotone in rarefaction
        return p_s

    # mirror u in wall frame; use p_star for energy
    rho_g = rho_i.copy()
    u_g   = 2.0*u_b - u_i
    v_g   = v_i
    p_star= _wall_star_pressure_exact(gamma, rho_i, p_i, u_i - u_b)
    E_g   = p_star/(gamma - 1.0) + 0.5 * rho_g * (u_g*u_g + v_g*v_g)

    # write requested variable into all ghost layers
    if variable == "density":
        val = rho_g
    elif variable == "x-momentum":
        val = rho_g * u_g
    elif variable == "y-momentum":
        val = rho_g * v_g
    else:  # "energy"
        val = E_g

    for i in range(ilo - 1, ilo - ng - 1, -1):
        ccdata.get_var(variable)[i, :] = val


def user_inflow(bc_name, bc_edge, variable, ccdata):
    """
    LEFT boundary, subsonic inflow:
      - prescribe u_b = piston_bc(t)
      - take J^- from interior first cell
      - keep K = p/rho^gamma from interior (isentropic inflow)
      - fill ghost layers for the requested variable only
    """
    if bc_name != "piston" or bc_edge != "xlb":
        return
    if variable not in ("density", "x-momentum", "y-momentum", "energy"):
        return

    # Step 1: Extract the grid information
    myg = ccdata.grid
    ilo, ng = myg.ilo, myg.ng

    # Step 2: Extract simulation parameters and specific heat ratio (gamma)
    gamma = float(ccdata.get_aux("gamma"))
    cp = float(ccdata.get_aux("cp"))
    cv = cp / gamma
    R = cp - cv  # = cp*(gamma-1)/gamma

    # Step 2: Calculate the piston speed at the current time
    tnow = float(getattr(ccdata, "t", 0.0))
    u_b = piston_bc(
        tnow,
        kind=ccdata.get_aux("piston.kind"),
        U=ccdata.get_aux("piston.initialVelocity"),
        A=ccdata.get_aux("piston.amplitude"),
        f=ccdata.get_aux("piston.frequency"),
        ramp_time=ccdata.get_aux("piston.rampTime"),
    )

    # Step 3: Calculate the interior isentropic relationships
    # Interior primitives at i = ilo
    rho_i = np.maximum(ccdata.get_var("density")[ilo, :], 1e-20)
    u_i = ccdata.get_var("x-momentum")[ilo, :] / rho_i
    v_i = ccdata.get_var("y-momentum")[ilo, :] / rho_i
    Et_i = ccdata.get_var("energy")[ilo, :]
    p_i = (gamma - 1.0) * np.maximum(Et_i - 0.5 * rho_i * (u_i * u_i + v_i * v_i), 0.0)

    # Static T, a, M in the interior
    T_i = np.maximum(p_i / (rho_i * R), 1e-6)
    a_i = np.sqrt(np.maximum(gamma * p_i / rho_i, 0.0))
    M_i = np.sqrt(u_i * u_i + v_i * v_i) / np.maximum(a_i, 1e-30)

    # Stagnation (isentropic) from interior
    fac_i = 1.0 + 0.5 * (gamma - 1.0) * M_i * M_i  # 1 + (γ-1) M^2 / 2
    T0 = T_i * fac_i
    p0 = p_i * np.power(fac_i, gamma / (gamma - 1.0))
    rho0 = rho_i * np.power(fac_i, 1.0 / (gamma - 1.0))
    h0 = cp * T0  # also = e + p/ρ + u^2/2

    # Step 4: Calculate the boundary cells state using isentropic relations
    v_b = v_i  # Keep the tangential velocity component
    T_b = T0 - 0.5 * (u_b * u_b + v_i * v_i) / cp # Static T at the boundary from h0 = cp*T0 = cp*T_b + u_b^2/2 + v_i^2/2
    p_b = np.maximum(p0 * np.power(np.maximum(T_b / T0, 1e-30), gamma / (gamma - 1.0)), 0.0)
    rho_b = np.maximum(rho0 * np.power(np.maximum(T_b / T0, 1e-30), 1.0 / (gamma - 1.0)), 1e-20)
    Et_b = rho_b * (p_b / ((gamma - 1.0) * rho_b)) + 0.5 * rho_b * (u_b*u_b + v_i*v_i)

    if variable == "density":
        val = rho_b
    elif variable == "x-momentum":
        val = rho_b * u_b
    elif variable == "y-momentum":
        val = rho_b * v_b
    else:  # "energy"
        val = Et_b

    for i in range(ilo - 1, ilo - ng - 1, -1):
        ccdata.get_var(variable)[i, :] = val


def init_data(my_data, rp):
    if rp.get_param("driver.verbose"):
        msg.bold("initializing the accelerating piston problem...")

    # -----------------------------
    # Far-field initial conditions
    # -----------------------------
    rho0  = float(rp.get_param("ic.density"))
    u0    = float(rp.get_param("ic.velocity"))
    p0    = float(rp.get_param("ic.pressure"))
    gamma = float(rp.get_param("eos.gamma"))

    # Required piston inputs
    u_p   = float(rp.params.get("piston.initialVelocity"))
    smooth_flag = int(rp.params.get("piston.initialSmoothing", 0))
    bin_size    = int(rp.params.get("piston.initialBinSize", 0))  # in cells

    # Field handles
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # Simulation Constants
    my_data.set_aux("cp", float(rp.get_param("eos.cp")))
    my_data.set_aux("cv", float(rp.get_param("eos.cp")))

    # Piston Parameters
    my_data.set_aux("piston.kind", rp.get_param("piston.kind"))
    my_data.set_aux("piston.initialVelocity", rp.get_param("piston.initialVelocity"))
    my_data.set_aux("piston.amplitude", rp.get_param("piston.amplitude"))
    my_data.set_aux("piston.frequency", rp.get_param("piston.frequency"))
    my_data.set_aux("piston.rampTime", rp.get_param("piston.rampTime"))

    # -----------------------------
    # Uniform far-field fill
    # -----------------------------
    dens[:] = rho0
    xmom[:] = rho0 * u0
    ymom[:] = 0.0
    ener[:] = p0 / (gamma - 1.0) + 0.5 * rho0 * (u0**2)

    # Grid info
    g    = my_data.grid
    ilo  = g.ilo            # first interior i
    ihi  = g.ihi            # last  interior i

    # Precompute far-field invariants
    K  = p0 / (max(rho0, RHO_FLOOR)**gamma)
    a0 = np.sqrt(max(gamma * p0 / max(rho0, RHO_FLOOR), 0.0))
    a0 = max(a0, A_FLOOR)

    if smooth_flag == 1 and bin_size > 0:
        # ---------------------------------------------
        # C¹ smoothstep from u_p (at wall) -> u0 (bulk)
        # over 'bin_size' interior cells starting at ilo
        # Ensures du/dx = 0 at i_start (ilo) and i_end
        # ---------------------------------------------
        i_start = ilo + 2
        i_end = min(ilo + bin_size - 1, ihi)
        if i_end >= i_start:
            n_cols = i_end - i_start + 1

            # normalized coordinate s in [0,1] from piston (s=0) to bulk (s=1)
            s = np.linspace(0.0, 1.0, n_cols, endpoint=True)[:, None]  # (n_cols, 1) broadcasts over y

            # cubic smoothstep S(s) = 3s^2 - 2s^3, S'(0)=S'(1)=0
            S = 3.0 * s ** 2 - 2.0 * s ** 3
            w = 1.0 - S  # weight: 1 at wall, 0 in bulk

            # velocity profile with zero slope at both ends
            u_prof = w * u_p + (1.0 - w) * u0

            # Isentropic state consistent with u_prof and far-field J^-
            a_prof = a0 + 0.5 * (gamma - 1.0) * (u_prof - u0)
            a_prof = np.maximum(a_prof, A_FLOOR)

            rho_prof = ((a_prof * a_prof) / (gamma * max(K, P_FLOOR))) ** (1.0 / (gamma - 1.0))
            rho_prof = np.maximum(rho_prof, RHO_FLOOR)

            p_prof = K * (rho_prof ** gamma)

            # Write the smoothed strip for all y
            jj = slice(None)
            dens[i_start:i_end + 1, jj] = rho_prof
            xmom[i_start:i_end + 1, jj] = rho_prof * u_prof
            ymom[i_start:i_end + 1, jj] = 0.0
            ener[i_start:i_end + 1, jj] = p_prof / (gamma - 1.0) + 0.5 * rho_prof * (u_prof ** 2)

            pad = min(2, ihi - ilo)
            dens[ilo:ilo + pad + 1, :] = rho_prof[0, :]
            xmom[ilo:ilo + pad + 1, :] = rho_prof[0, :] * u_prof[0, :]
            ymom[ilo:ilo + pad + 1, :] = 0.0
            ener[ilo:ilo + pad + 1, :] = p_prof[0, :] / (gamma - 1.0) + 0.5 * rho_prof[0, :] * (u_prof[0, :] ** 2)
        # else: bin_size range outside domain -> no-op
    else:
        # ---------------------------------------------
        # No smoothing: set a single column to piston state
        # ---------------------------------------------
        i = ilo  # first interior column
        a1   = a0 + 0.5 * (gamma - 1.0) * (u_p - u0)
        a1   = max(a1, A_FLOOR)
        rho1 = ((a1*a1) / (gamma * max(K, P_FLOOR))) ** (1.0 / (gamma - 1.0))
        rho1 = max(rho1, RHO_FLOOR)
        p1   = K * (rho1 ** gamma)

        dens[i, :] = rho1
        xmom[i, :] = rho1 * u_p
        ymom[i, :] = 0.0
        ener[i, :] = p1 / (gamma - 1.0) + 0.5 * rho1 * (u_p*u_p)


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
        #nitrogenAmount=0.5 * 3.76,
        mech='../Li-Dryer-H2-mechanism.yaml',
    )  # ✅ shared state

    # Step 3: Initialize the pyroSolver with the gas object and output directory
    simulation = pyroSolver(
        sim_time=8.5e-5,  # total simulation time in seconds
        cfl=0.001,  # CFL number
        nx=10240,  # number of grid points in x-direction
        ny=8,  # number of grid points in y-direction
        xmax=0.001,  # maximum x-coordinate
        ymax=0.0005,  # maximum y-coordinate
        bc={"xl": "piston", "xr": "outflow", "yl": "periodic", "yr": "periodic"},  # boundary conditions
        piston={'Initial Smoothing': 0,  # 0: no smoothing, 1: tanh smoothing
                'Initial Bin Size': 12,},
        output_dir='Oscillating-Piston-Data-Animation-Frames'  # directory to save output files
    )

    simulation.setup(piston_mode="inflow")
    simulation.run(time_stepping=True)

    return

if __name__ == "__main__":
    main()