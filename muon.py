import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

np.random.seed(42) #set random seed

# PDF & inverse‐CDF for θ ∈ [–π/2, π/2]
norm_const = 2/np.pi
P_theta   = lambda th: np.cos(th)**2 * norm_const

grid = np.linspace(-np.pi/2, np.pi/2, 10_000)
cdf_vals = np.array([quad(P_theta, -np.pi/2, t)[0] for t in grid])
inverse_cdf = interp1d(cdf_vals, grid, bounds_error=False,
                       fill_value=(-np.pi/2, np.pi/2))

# PARAMETERS SETUP
N        = 100_000
z_values = [0.0, 0.5, 1.0, 1.25, 2.0]
side     = 1.0
half     = side/2

# Draw angles
u         = np.random.rand(N)
theta     = inverse_cdf(u)
phi       = 2*np.pi*np.random.rand(N)
theta_deg = np.degrees(theta)

# Precompute the z=0 histogram
bins       = np.linspace(-90, 90, 91)
gen_counts, _ = np.histogram(theta_deg, bins=bins)

# ----------------------------
# 3D Plot of Particle Trajectories (for a single z value)
# ----------------------------
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# 3D Plot
z_3d = 1.0 # Set any distance you want
r3d = z_3d / np.cos(theta)
x0_3d = np.random.uniform(-half, half, 100)  # fewer for clarity
y0_3d = np.random.uniform(-half, half, 100)
phi_3d = 2*np.pi*np.random.rand(100)
theta_3d = inverse_cdf(np.random.rand(100))

dx3d = r3d[:100] * np.sin(theta_3d) * np.cos(phi_3d)
dy3d = r3d[:100] * np.sin(theta_3d) * np.sin(phi_3d)
xh3d = x0_3d + dx3d
yh3d = y0_3d + dy3d

fig3d = plt.figure(figsize=(10, 8), facecolor='white')
ax3d = fig3d.add_subplot(111, projection='3d')

# Draw the source (z=0) and target (z=-z_3d) square boundaries
def draw_square(ax, z_val, color, label=None):
    corners = [
        [-half, -half, z_val],
        [ half, -half, z_val],
        [ half,  half, z_val],
        [-half,  half, z_val],
        [-half, -half, z_val]
    ]
    xs, ys, zs = zip(*corners)
    ax.plot(xs, ys, zs, color=color, alpha=0.4, label=label)

draw_square(ax3d, 0, 'blue', label='Source plane (z=0)')
draw_square(ax3d, -z_3d, 'red', label=f'Target plane (z=-{z_3d} m)')

# Plot a few trajectories
for i in range(100):
    ax3d.plot([x0_3d[i], xh3d[i]], [y0_3d[i], yh3d[i]], [0, -z_3d], 
              color='black', alpha=0.3, lw=0.5)

ax3d.set_xlabel("X [m]")
ax3d.set_ylabel("Y [m]")
ax3d.set_zlabel("Z [m]")
ax3d.set_title(f"3D Particle Trajectories (z = {z_3d} m)")
ax3d.legend()
ax3d.view_init(elev=25, azim=45)

# ----------------------------
# 2D Histogram of Impact Positions at z = 1.0 m
# ----------------------------
z_target = 1.0
r   = z_target / np.cos(theta)
x0  = np.random.uniform(-half, half, N)
y0  = np.random.uniform(-half, half, N)
dx  = r * np.sin(theta) * np.cos(phi)
dy  = r * np.sin(theta) * np.sin(phi)
xh  = x0 + dx
yh  = y0 + dy

inside = (
    (xh >= -half) & (xh <= half) &
    (yh >= -half) & (yh <= half)
)

# Plot 2D histogram for accepted particles only
plt.figure(figsize=(8, 6), facecolor='white')
plt.hist2d(xh[inside], yh[inside], bins=100, cmap='viridis')
plt.colorbar(label='Counts')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('2D Impact Position Histogram on Bottom Detector (z = 1.0 m)')
plt.grid(False)
plt.axis('equal')
plt.tight_layout()
plt.show()

# ----------------------------
# Plot the θ Distributions
# ----------------------------

fig, (ax_gen, ax) = plt.subplots(2, 1, figsize=(8, 12), facecolor='white', sharex=True)

# ----------------------------
# Plot the Generated θ Distribution (Top subplot)
# ----------------------------
ax_gen.hist(theta_deg, bins=bins, histtype='step', linewidth=1.5, label='Generated θ')
ax_gen.set_ylabel('Counts')
ax_gen.set_title('Generated Theta Distribution (Top Detector)')
ax_gen.legend()
ax_gen.grid(True)

# ----------------------------
# Plot the Measured θ Distribution (Bottom subplot)
# ----------------------------
accepted_events = {}

for z in z_values:
    r   = z / np.cos(theta)
    x0  = np.random.uniform(-half, half, N)
    y0  = np.random.uniform(-half, half, N)
    dx  = r * np.sin(theta) * np.cos(phi)
    dy  = r * np.sin(theta) * np.sin(phi)
    xh  = x0 + dx
    yh  = y0 + dy

    inside = (
        (xh >= -half) & (xh <= half) &
        (yh >= -half) & (yh <= half)
    )

    acc_counts, _ = np.histogram(theta_deg[inside], bins=bins)
    weights_per_bin = gen_counts / np.maximum(acc_counts, 1)
    bin_idx = np.digitize(theta_deg[inside], bins) - 1
    evt_weights = weights_per_bin[bin_idx]

    accepted_events[z] = round((len(theta_deg[inside]) / N) * 100, 3)

    ax.hist(
        theta_deg[inside],
        bins=bins,
        weights=evt_weights,
        histtype='step',
        linewidth=1.5,
        label=f'z = {z:.2f} m (ACCEPTANCE: {accepted_events[z]}%)'
    )

ax.set_xlabel('θ (degrees)')
ax.set_ylabel('Counts')
ax.set_title('Theta Distribution for Different Trigger Distances (Bottom Detector)')
ax.legend()
ax.grid(True)


# ----------------------------
# Plot the φ Distribution
# ----------------------------

fig_phi, (ax_phi_gen, ax_phi_meas) = plt.subplots(2, 1, figsize=(8, 12), facecolor='white', sharex=True)

# ----------------------------
# Plot the Generated φ Distribution (Top subplot)
# ----------------------------
phi_deg = np.degrees(phi)
phi_bins = np.linspace(0, 360, 181)
gen_counts_phi, _ = np.histogram(phi_deg, bins=phi_bins)

ax_phi_gen.hist(
    phi_deg,
    bins=phi_bins,
    histtype='step',
    linewidth=1.5,
    label='Generated φ'
)
ax_phi_gen.set_ylabel('Counts')
ax_phi_gen.set_title('Generated Phi Distribution (Top Detector)')
ax_phi_gen.legend()
ax_phi_gen.grid(True)

# ----------------------------
# Plot the Generated φ Distribution (Bottom subplot)
# ----------------------------
for z in z_values:
    r   = z / np.cos(theta)
    x0  = np.random.uniform(-half, half, N)
    y0  = np.random.uniform(-half, half, N)
    dx  = r * np.sin(theta) * np.cos(phi)
    dy  = r * np.sin(theta) * np.sin(phi)
    xh  = x0 + dx
    yh  = y0 + dy

    inside = (
        (xh >= -half) & (xh <= half) &
        (yh >= -half) & (yh <= half)
    )

    phi_inside_deg = phi_deg[inside]
    acc_counts_phi, _ = np.histogram(phi_inside_deg, bins=phi_bins)
    weights_per_bin_phi = gen_counts_phi / np.maximum(acc_counts_phi, 1)
    bin_idx_phi = np.digitize(phi_inside_deg, bins=phi_bins) - 1
    bin_idx_phi = np.clip(bin_idx_phi, 0, len(weights_per_bin_phi) - 1)
    evt_weights_phi = weights_per_bin_phi[bin_idx_phi]

    ax_phi_meas.hist(
        phi_inside_deg,
        bins=phi_bins,
        weights=evt_weights_phi,
        histtype='step',
        linewidth=1.5,
        label=f'z = {z:.2f} m'
    )

ax_phi_meas.set_xlabel('φ (degrees)')
ax_phi_meas.set_ylabel('Counts')
ax_phi_meas.set_title('Phi Distribution for Different Trigger Distances (Bottom Detector)')
ax_phi_meas.legend()
ax_phi_meas.grid(True)

plt.tight_layout()
plt.show()
