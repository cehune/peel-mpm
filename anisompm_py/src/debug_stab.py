import sys; sys.path.insert(0, "src")
import torch
from orange_setup import prepare_orange
from anisompm import AnisoMPM, halfspace_collider

DEV = "cuda:1"
PLY = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.ply"
data = prepare_orange(PLY, n_grid=64, parts_per_cell_dim=2.0, splat_sigma=0.006, device=DEV)
lo = data["sim_x0"].min(0).values.cpu().numpy()
hi = data["sim_x0"].max(0).values.cpu().numpy()
print("orange sim bbox", lo, hi, flush=True)


def mk(E, grav, dt, damp=1.0, ground=False, dx=None):
    sim = AnisoMPM(n_grid=64, grid_lim=1.0, dt=dt, gravity=(0, grav, 0), grid_damp=damp, device=DEV)
    sim.add_object(data["sim_x0"], data["sim_vol"], rho=500, E=E, nu=0.4,
                   fibers=None, alpha=0.0, allow_damage=False)
    if ground:
        sim.colliders.append(halfspace_collider(
            lambda t: [0, float(lo[1]) + 0.5 * sim.dx, 0], [0, 1, 0], mode="slip", friction=0.6))
    return sim


def report(tag, sim, nsub, nblk=6):
    for f in range(nblk):
        sim.run_frame(nsub)
        S = torch.linalg.svdvals(sim.F)
        nan = bool(torch.isnan(sim.x).any())
        print(f"  {tag} blk{f} Smax={float(S.max()):.3f} Smin={float(S.min()):.3f}"
              f" |v|max={float(sim.v.norm(dim=1).max()):.3f} ymin={float(sim.x[:,1].min()):.3f} nan={nan}", flush=True)
        if nan:
            break


print("\nREST  E=1e4 dt=3e-4 (expect no motion):", flush=True)
report("rest", mk(1e4, 0.0, 3e-4), 60)
print("\nGRAVITY only  E=1e4 dt=3e-4 grav=-2 (free fall, rigid):", flush=True)
report("grav", mk(1e4, -2.0, 3e-4), 60)
print("\nGRAVITY+GROUND  E=1e4 dt=3e-4 grav=-2 damp=.999:", flush=True)
report("grnd", mk(1e4, -2.0, 3e-4, damp=0.999, ground=True), 60)
print("\nGRAVITY+GROUND  E=3e4 dt=2e-4 grav=-2 damp=.999:", flush=True)
report("grnd2", mk(3e4, -2.0, 2e-4, damp=0.999, ground=True), 84)
print("\ndone", flush=True)
