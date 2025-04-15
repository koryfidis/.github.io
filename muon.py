import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ----------------------------
# Physics Setup
# ----------------------------

#Number of events
N = 5000
#Set surface distance
z=-0.5

# Normalized PDF: P(θ) = (2/π) cos²(θ) [radians]
norm_const = 2 / np.pi
P_theta = lambda theta: (np.cos(theta))**2 * norm_const

# Generate θ values using inverse transform sampling
theta_grid_rad = np.linspace(-np.pi/2, np.pi/2, N)
cdf_vals = np.array([quad(P_theta, -np.pi/2, theta)[0] for theta in theta_grid_rad])
inverse_cdf = interp1d(cdf_vals, theta_grid_rad, kind='linear', bounds_error=False, fill_value=(-np.pi/2, np.pi/2))


u_vals = np.random.uniform(0, 1, N)
theta_vals_rad = inverse_cdf(u_vals)

# Convert to degrees
theta_vals_deg = np.degrees(theta_vals_rad)
theta_grid_deg = np.degrees(theta_grid_rad)

# Generate φ values (0-360 degrees)
phi_vals_rad = np.random.uniform(0, 2*np.pi, N)
phi_vals_deg = np.degrees(phi_vals_rad) % 360  # Ensure [0, 360)

# ----------------------------
# Trajectory Simulation
# ----------------------------
x_start = np.random.uniform(-0.1,0.1, N)
y_start = np.random.uniform(-0.1,0.1, N)
r = np.abs(z) * np.tan(theta_vals_rad) 

x_final = x_start + r * np.cos(phi_vals_rad)
y_final = y_start + r * np.sin(phi_vals_rad)

# Filter hits on target [0,1]x[0,1]
mask = (x_final >= -0.5) & (x_final <= 0.5) & (y_final >= -0.5) & (y_final <= 0.5)
theta_hit_deg = theta_vals_deg[mask]
phi_hit_deg = phi_vals_deg[mask]


# Header
print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "x_start", "y_start", "θ_start", "φ_start", 
    "x_final", "y_final", "z_final", "θ_final", "φ_final"
))

print("="*100)

# Print first 100 tracks
for i in range(min(100, sum(mask))):
    idx = np.where(mask)[0][i]  # Get the index of the i-th hit
    print("{:<10.4f} {:<10.4f} {:<10.2f} {:<10.2f} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.2f} {:<10.2f}".format(
        x_start[idx], y_start[idx], 
        theta_vals_deg[idx], phi_vals_deg[idx],
        x_final[idx], y_final[idx], z,
        theta_vals_deg[idx], phi_vals_deg[idx]
    ))

# Summary statistics
print("\nSummary Statistics for Hits:")
print(f"Total particles: {N}")
print(f"Particles that hit the surface: {sum(mask)} ({sum(mask)/N*100:.2f}%)")
print(f"Average θ of hits: {np.mean(theta_hit_deg):.2f}°")
print(f"Average φ of hits: {np.mean(phi_hit_deg):.2f}°")


# For θ hits (Plot 5)
counts_hit_theta, bins_hit_theta = np.histogram(theta_hit_deg, bins=72, range=(-90, 90))
bin_centers_hit_theta = (bins_hit_theta[:-1] + bins_hit_theta[1:])/2  #bin center
np.savetxt('theta_hit_bins.txt', np.column_stack((bin_centers_hit_theta, counts_hit_theta)), 
           header="Bin_Center[deg] Count", fmt="%.2f %d")

# For φ hits (Plot 6)
counts_hit_phi, bins_hit_phi = np.histogram(phi_hit_deg, bins=360, range=(0, 360))
bin_centers_hit_phi = (bins_hit_phi[:-1] + bins_hit_phi[1:])/2
np.savetxt('phi_hit_bins.txt', np.column_stack((bin_centers_hit_phi, counts_hit_phi)), 
          header="Bin_Center[deg] Count", fmt="%.2f %d")

# Export distributions in degrees/ For ROOT
np.savetxt('theta_hit_deg.txt', theta_hit_deg)
np.savetxt('phi_hit_deg.txt', phi_hit_deg)

# ----------------------------
# Plotting (All in Degrees)
# ----------------------------
fig = plt.figure(figsize=(18, 20))
ax1 = fig.add_subplot(321, projection='3d')
ax1.plot([0, 1], [0, 0], [0, 0], 'b-', alpha=0.3, label="Source (z=0)")
ax1.plot([0, 1], [0, 0], [-1, -1], 'r-', alpha=0.3, label="Target (z=-1m)")
ax1.plot([0, 0], [0, 1], [0, 0], 'b-', alpha=0.3)
ax1.plot([0, 0], [0, 1], [-1, -1], 'r-', alpha=0.3)
ax1.plot([0, 1], [1, 1], [0, 0], 'b-', alpha=0.3)
ax1.plot([1, 0], [1, 1], [-1, -1], 'r-', alpha=0.3)
ax1.plot([1, 1], [0, 1], [0, 0], 'b-', alpha=0.3)
ax1.plot([1, 1], [0, 1], [-1, -1], 'r-', alpha=0.3)
for i in range(100):
    ax1.plot([x_start[i], x_final[i]], [y_start[i], y_final[i]], [0, -1.0], 
             'k-', alpha=0.3, lw=0.5)
ax1.view_init(elev=25, azim=45)
ax1.set_xlabel("X [m]")
ax1.set_ylabel("Y [m]")
ax1.set_zlabel("Z [m]")
ax1.set_title("3D Particle Trajectories")

# Plot 2: XY Positions
ax2 = fig.add_subplot(322)
ax2.scatter(x_start, y_start, color='b', alpha=0.3, s=2, 
            label="Initial Positions (z=0)")
ax2.scatter(x_final[mask], y_final[mask], color='r', alpha=0.3, s=2, 
            label="Successful Hits (z=-1m)")
ax2.set_xlabel("X [m]")
ax2.set_ylabel("Y [m]")
ax2.set_title("Position Distributions at Surfaces")
ax2.legend()



# Plot 3: Theta Distribution (Generator)
ax3 = fig.add_subplot(323)
counts_all_theta, bins, _ = ax3.hist(theta_vals_deg, bins=72, density=True, alpha=0.6, color='green',
                                   label="Simulated θ (All)")
total_all_theta = len(theta_vals_deg)  # Total particles 
ax3.plot(theta_grid_deg, P_theta(theta_grid_rad) * (np.pi/180), 'r-', 
        label="Theory: (2/π)cos²θ [Scaled]")
ax3.text(0.95, 0.95, f'Total: {total_all_theta}', transform=ax3.transAxes, 
        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
ax3.set_xlabel("θ [degrees]")
ax3.set_ylabel("Frequency")
ax3.set_title("Angular Distribution - Generated Particles")
ax3.legend()

# Plot 4: Phi Distribution (Generator)
ax4 = fig.add_subplot(324)
counts_all_phi, bins, _ = ax4.hist(phi_vals_deg, bins=361, density=True, alpha=0.6, color='purple',
                                 label="Simulated φ (All)")
total_all_phi = len(phi_vals_deg)  # Same as N
ax4.axhline(1/360, color='k', ls='--', label="Uniform (1/360°)")
ax4.text(0.95, 0.95, f'Total: {total_all_phi}', transform=ax4.transAxes,
        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
ax4.set_xlabel("φ [degrees]")
ax4.set_ylabel("Frequency")
ax4.set_title("Azimuthal Distribution - Generated Particles")
ax4.legend()

# Plot 5: Theta Distribution (Hits)
ax5 = fig.add_subplot(325)
counts_hit_theta, bins, _ = ax5.hist(theta_hit_deg, bins=72, density=True,alpha=0.6, color='green', label="Simulated θ (Hits)")
total_hit_theta = len(theta_hit_deg)  # Number of successful hits
ax5.plot(theta_grid_deg, P_theta(theta_grid_rad) * (np.pi/180), 'r-', 
        label="Theoretical Prediction")
ax5.text(0.95, 0.95, f'Total Hits: {total_hit_theta}', transform=ax5.transAxes,
        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
ax5.text(0.95, 0.85, f'Surface Distance: {z}m', transform=ax5.transAxes,
        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))        
ax5.set_xlabel("θ [degrees]")
ax5.set_ylabel("Frequency")
ax5.set_title("Angular Distribution - Successful Hits")
ax5.legend()

# Plot 6: Phi Distribution (Hits)
ax6 = fig.add_subplot(326)
counts_hit_phi, bins, _ = ax6.hist(phi_hit_deg, bins=361, density=True, alpha=0.6, color='purple', label="Simulated φ (Hits)")
total_hit_phi = len(phi_hit_deg)  # Same as number of hits
ax6.axhline(1/360, color='k', ls='--', label="Uniform (1/360°)")
ax6.text(0.95, 0.95, f'Total: {total_hit_phi}', transform=ax6.transAxes,
        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
ax6.text(0.95, 0.85, f'Surface Distance: {z}m', transform=ax6.transAxes,
        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))      
ax6.set_xlabel("φ [degrees]")
ax6.set_ylabel("Frequency")
ax6.set_title("Azimuthal Distribution - Successful Hits")
ax6.legend()

plt.figure(figsize=(10,8))
plt.hist2d(x_start, y_start, bins=100, range=[[0,1],[0,1]])
plt.colorbar(label='Hit density')
plt.title('Hit Position Distribution on Lower Surface')
plt.xlabel('X position')
plt.ylabel('Y position')

plt.figure(figsize=(10,8))
plt.hist2d(x_final[mask], y_final[mask], bins=100, range=[[0,1],[0,1]])
plt.colorbar(label='Hit density')
plt.title('Hit Position Distribution on Generator Surface')
plt.xlabel('X position')
plt.ylabel('Y position')

plt.tight_layout()
plt.show()
