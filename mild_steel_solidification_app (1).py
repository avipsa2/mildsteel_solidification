import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout='wide', page_title='Mild Steel Solidification Live Simulation')

MAX_SIM_TIME = 60.0
SNAP_INTERVAL = 0.2

DEFAULT_PARAMS = {
    'L': 0.01,
    'nx': 60,
    'ny': 60,
    'dt': 1e-4,
    'rho': 7850.0,
    'cp': 500.0,
    'k': 120.0,
    'L_latent': 272000.0,
    'T_liquidus': 1530.0,
    'T_solidus': 1495.0,
    'T_pour': 1550.0,
    'T_mold': 25.0,
    'mu': 0.006,
    'beta': 1e-4,
    'g': 9.81,
    'h_top': 8000.0
}

STAGGER_FACTOR = 50.0  

def make_params(overrides=None):
    p = DEFAULT_PARAMS.copy()
    if overrides:
        p.update(overrides)
    p['dx'] = p['L']/p['nx']
    p['dy'] = p['L']/p['ny']
    return p

def initialize_fields(p):
    nx, ny = p['nx'], p['ny']
    T = np.full((ny,nx), p['T_pour'], dtype=float)
    return {
        'T': T.copy(),
        'T_old': T.copy(),
        'fs': np.zeros_like(T),
        'fl': np.ones_like(T),
        'u': np.zeros_like(T),
        'v': np.zeros_like(T)
    }

# ---------------- PHYSICS ----------------
def calc_solid_fraction(T, p):
    Tl, Ts = p['T_liquidus'], p['T_solidus']
    return np.where(T>=Tl,0.0,np.where(T<=Ts,1.0,(Tl-T)/(Tl-Ts)))

def solve_heat_conduction(fields, p):
    T, T_old = fields['T'], fields['T_old']
    nx, ny, dx, dy, dt = p['nx'], p['ny'], p['dx'], p['dy'], p['dt']
    rho, cp, k, L_latent = p['rho'], p['cp'], p['k'], p['L_latent']

    T_ip = np.roll(T,-1,axis=1); T_im=np.roll(T,1,axis=1)
    T_jp = np.roll(T,-1,axis=0); T_jm=np.roll(T,1,axis=0)
    lap = (T_ip+T_im-2*T)/dx**2 + (T_jp+T_jm-2*T)/dy**2

    T_predict = T + (k/(rho*cp))*lap*dt
    fs_old = fields['fs']
    fs_new = calc_solid_fraction(T_predict,p)
    dfs = fs_new - fs_old
    denom = T_predict - T_old
    denom[np.abs(denom)<1e-12] = 1e-12

    dfs_dT = dfs/denom
    Ceff = np.maximum(cp + L_latent*dfs_dT, cp)

    dTdt = k*lap/(rho*Ceff)
    T_next = T + dTdt*dt

    # Top convective BC
    T_next[-1,:] = (k*T_next[-2,:]/dy + p['h_top']*p['T_mold'])/(k/dy + p['h_top'])
    # Adiabatic BCs
    T_next[:,0] = T_next[:,1]; T_next[:,-1] = T_next[:,-2]; T_next[0,:] = T_next[1,:]
    return T_next

def solve_thermal_buoyancy(fields,p):
    dt, mu, beta, g, Tliq = p['dt'], p['mu'], p['beta'], p['g'], p['T_liquidus']
    u, v = fields['u'].copy(), fields['v'].copy()
    fs, fl = fields['fs'], fields['fl']

    F_buoy = -beta*g*(fields['T']-Tliq)
    K_perm = 1e-10*(fl**3)/(fs**2 + 1e-12)
    S_darcy_u = np.where(fs>0.01, -mu*u/(K_perm+1e-20),0.0)
    S_darcy_v = np.where(fs>0.01, -mu*v/(K_perm+1e-20),0.0)

    u += dt*S_darcy_u*STAGGER_FACTOR
    v += dt*(F_buoy + S_darcy_v)*STAGGER_FACTOR

    u = np.clip(u,-0.02,0.02); v=np.clip(v,-0.05,0.05)
    u[0,:]=u[-1,:]=u[:,0]=u[:,-1]=0
    v[0,:]=v[-1,:]=v[:,0]=v[:,-1]=0
    return u, v

def simulation_step(state, case, p):
    state['T_old'] = state['T'].copy()
    state['T'] = solve_heat_conduction(state,p)
    
    if case>=3:
        u, v = solve_thermal_buoyancy(state,p)
        state['u'], state['v'] = u, v

    state['fs'] = calc_solid_fraction(state['T'],p)
    state['fl'] = 1.0 - state['fs']

    return state

st.title("Mild Steel Solidification Simulation")
st.sidebar.header("Simulation Controls")

case = st.sidebar.radio("Select Case",[1,2,3],
                        format_func=lambda x:{1:"Pure Heat",
                                             2:"Alloy",
                                             3:"Alloy + Flow"}[x])
nx_in = st.sidebar.number_input("Grid nx=ny", min_value=20,value=int(DEFAULT_PARAMS['nx']),step=10)
dt_in = st.sidebar.number_input("Time step dt (s)", min_value=1e-6,value=float(DEFAULT_PARAMS['dt']),format="%.6f")
h_top_in = st.sidebar.number_input("Top convective h (W/m²K)", min_value=100.0,value=float(DEFAULT_PARAMS['h_top']),step=100.0)

start_btn = st.sidebar.button("Start Simulation")
stop_btn = st.sidebar.button("Stop Simulation")
reset_btn = st.sidebar.button("Reset Simulation")

user_overrides = {'nx':int(nx_in),'ny':int(nx_in),'dt':float(dt_in),'h_top':float(h_top_in)}
p = make_params(user_overrides)

if 'initialized' not in st.session_state or reset_btn:
    st.session_state.params = p.copy()
    st.session_state.state = initialize_fields(p)
    st.session_state.time = 0.0
    st.session_state.is_running = False
    st.session_state.initialized = True

state = st.session_state.state

# CFL display
alpha = p['k']/(p['rho']*p['cp'])
cfl = alpha*p['dt']/p['dx']**2
st.sidebar.text(f"CFL = {cfl:.6f} (<0.25 safe)")

display_area = st.empty()

# simulation
if start_btn: st.session_state.is_running = True
if stop_btn: st.session_state.is_running = False

if st.session_state.is_running:
    while st.session_state.time<MAX_SIM_TIME and st.session_state.is_running:
        steps_per_snap = max(1,int(SNAP_INTERVAL/p['dt']))
        for _ in range(steps_per_snap):
            state = simulation_step(state,case,p)
            st.session_state.time += p['dt']

        fig, axes = plt.subplots(1,2,figsize=(12,6))
        ax1, ax2 = axes.flatten()

        im1 = ax1.imshow(state['T'],origin='lower',
                         extent=[0,p['L']*1000,0,p['L']*1000],
                         cmap='turbo',vmin=p['T_mold'],vmax=p['T_pour'])
        ax1.set_title("Temperature (°C)")
        fig.colorbar(im1,ax=ax1)

        if case>=3:
            skip = max(1,p['nx']//15)
            X,Y = np.meshgrid(np.linspace(0,p['L']*1000,p['nx']),
                              np.linspace(0,p['L']*1000,p['ny']))
            ax2.quiver(X[::skip,::skip],Y[::skip,::skip],
                       state['u'][::skip,::skip]*50,
                       state['v'][::skip,::skip]*50,
                       angles='xy',scale_units='xy',scale=1.0,color='blue')
            ax2.set_title("Velocity Field")
        else:
            ax2.text(0.5,0.5,"No Flow",ha='center',va='center')
            ax2.set_title("Velocity")

        plt.tight_layout()
        display_area.pyplot(fig)
        plt.close(fig)
        time.sleep(0.1)

st.session_state.state = state
