"""Head-to-head demo: 3 canonical warp drives + NEC and exotic-matter audit."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warp_drive_zoo import (
    AlcubierreDrive,
    BobrickMartireDrive,
    LentzDrive,
    compare_drives,
)


def main():
    drives = [
        AlcubierreDrive(v_s=1.0, R=1.0, sigma=8.0),
        BobrickMartireDrive(m_ADM=1.0, R=1.0, sigma=4.0,
                              name="bobrick_martire_pos"),
        BobrickMartireDrive(m_ADM=-1.0, R=1.0, sigma=4.0,
                              name="bobrick_martire_neg"),
        LentzDrive(v_s=0.7, sigma=4.0),
    ]
    cmp_ = compare_drives(drives, box_half_size=2.5, n_grid=20)

    print(f"{'drive':<28} {'wall (x,rho)':<18} {'NEC':>10} {'T_tt(wall)':>14} "
          f"{'|exotic E|':>14}  NEC_viol?")
    for d in drives:
        x_w, r_w = d.wall_location()
        nec = cmp_.nec_at_wall[d.name]
        rho = cmp_.energy_density_at_wall[d.name]
        ex = abs(cmp_.total_negative_energy[d.name])
        viol = cmp_.nec_violated[d.name]
        print(f"{d.name:<28} ({x_w:.2f},{r_w:.2f})    "
              f"{nec:>+10.4f} {rho:>+14.4e} {ex:>14.4e}  {viol}")

    print()
    print("Ranking by |total negative energy| (less exotic first):")
    for name, val in cmp_.ranking_by_exotic_matter:
        print(f"  {name:<28} |E_neg| = {val:.4e}")


if __name__ == "__main__":
    main()
