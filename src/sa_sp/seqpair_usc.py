#!/usr/bin/env python3
"""
Sequence-Pair Floorplanner with Simulated Annealing
Area & Wirelength Co-optimization

Optimizations over original:
  1. sp_to_placement: NumPy vectorized inner loops (was O(n²) pure Python)
  2. perturb: only copies the structure that actually changes (was 3 full copies every call)
  3. compute_hpwl: NumPy-based net bounding boxes
  4. compute_bounding_box: NumPy max
  5. SA schedule: MOVES_PER_TEMP changed from 10*n² to 20*n (linear); cooling
     factor relaxed slightly to compensate
"""

import argparse
import math
import random
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
# AD-> Reads the file to be tested on
def parse_input(filename):
    blocks = {}   # name -> (w, h)
    nets = []     # list of list of block names

    with open(filename) as f:
        section = None
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower() == 'blocks':
                section = 'blocks'
            elif line.lower() == 'nets':
                section = 'nets'
            elif section == 'blocks':
                parts = line.split()
                blocks[parts[0]] = (float(parts[1]), float(parts[2]))
            elif section == 'nets':
                nets.append(line.split())
    return blocks, nets

# AD-> Creates the blocks in use and rondomizes placement
def write_random_design(filename, num_blocks=10, num_nets=10, seed=42):
    rng = random.Random(seed)
    block_names = [f"B{i}" for i in range(num_blocks)]
    with open(filename, 'w') as f:
        f.write("# Auto-generated design\n")
        f.write("BLOCKS\n")
        for name in block_names:
            w = rng.randint(5, 30)
            h = rng.randint(5, 30)
            f.write(f"{name} {w} {h}\n")
        f.write("NETS\n")
        for _ in range(num_nets):
            k = rng.randint(2, 4)
            members = rng.sample(block_names, min(k, num_blocks))
            f.write(" ".join(members) + "\n")
    print(f"[INFO] Random design written to {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Pair → Placement (longest-path in constraint graph)
# ─────────────────────────────────────────────────────────────────────────────
# AD-> Takes the sequences and creates the floorplan
def sp_to_placement(sp_pos, sp_neg, blocks, block_names):
    """
    Given Γ+ and Γ- (lists of block indices), compute (x, y, w, h) for each
    block using the standard sequence-pair longest-path algorithm.

    Constraint rules (for blocks i and j):
      • pos_in_pos[i] < pos_in_pos[j] AND pos_in_neg[i] < pos_in_neg[j]
            → i is to the LEFT  of j  (x[j] >= x[i] + width[i])
      • pos_in_pos[i] < pos_in_pos[j] AND pos_in_neg[i] > pos_in_neg[j]
            → i is BELOW j            (y[j] >= y[i] + height[i])

    OPTIMIZATION: inner v-loop replaced with NumPy vectorized masks,
    reducing O(n²) pure-Python iterations to O(n) NumPy ops per outer step.
    """
    n = len(block_names)
    widths  = np.array([blocks[block_names[i]][0] for i in range(n)], dtype=float)
    heights = np.array([blocks[block_names[i]][1] for i in range(n)], dtype=float)

    pos_in_pos = np.empty(n, dtype=np.intp)
    pos_in_neg = np.empty(n, dtype=np.intp)
    sp_pos_arr = np.asarray(sp_pos, dtype=np.intp)
    sp_neg_arr = np.asarray(sp_neg, dtype=np.intp)
    pos_in_pos[sp_pos_arr] = np.arange(n, dtype=np.intp)
    pos_in_neg[sp_neg_arr] = np.arange(n, dtype=np.intp)

    # ── X: longest-path on the horizontal constraint DAG ─────────────
    x = np.zeros(n, dtype=float)
    for u in sp_pos_arr:                        # topological order = Γ+ order
        pip_u = pos_in_pos[u]
        pin_u = pos_in_neg[u]
        # all v where u is strictly left of v in both sequences
        mask = (pip_u < pos_in_pos) & (pin_u < pos_in_neg)
        mask[u] = False
        cand = x[u] + widths[u]
        np.maximum(x, cand, where=mask, out=x)

    # ── Y: longest-path on the vertical constraint DAG ───────────────
    y = np.zeros(n, dtype=float)
    for u in sp_pos_arr:
        pip_u = pos_in_pos[u]
        pin_u = pos_in_neg[u]
        # all v where u is below v (same Γ+ order, reversed Γ- order)
        mask = (pip_u < pos_in_pos) & (pin_u > pos_in_neg)
        mask[u] = False
        cand = y[u] + heights[u]
        np.maximum(y, cand, where=mask, out=y)

    return x, y, widths, heights

# AD-> Validates the floorplan for no blocks being over each other
def check_no_overlaps(x, y, widths, heights, block_names, tol=1e-6):
    """Raise AssertionError if any two blocks overlap (beyond floating-point tol)."""
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            x_overlap = (x[i] + widths[i]  - tol > x[j]) and (x[j] + widths[j]  - tol > x[i])
            y_overlap = (y[i] + heights[i] - tol > y[j]) and (y[j] + heights[j] - tol > y[i])
            if x_overlap and y_overlap:
                raise AssertionError(
                    f"Overlap: {block_names[i]} "
                    f"({x[i]:.1f},{y[i]:.1f},{widths[i]:.0f}x{heights[i]:.0f}) ∩ "
                    f"{block_names[j]} ({x[j]:.1f},{y[j]:.1f},{widths[j]:.0f}x{heights[j]:.0f})")

# AD-> Calcultes the smalles rectangle anchored at origin
def compute_bounding_box(x, y, widths, heights):
    if len(x) == 0:
        return 0.0, 0.0
    W = float(np.max(x + widths))
    H = float(np.max(y + heights))
    return W, H

#AD-> It calculates wirelength
def compute_hpwl(x, y, nets, name_to_idx):
    """
    Half-perimeter wirelength using lower-left corners.
    OPTIMIZATION: vectorized per-net min/max instead of Python list appends.
    """
    total = 0.0
    for net in nets:
        idxs = [name_to_idx[b] for b in net if b in name_to_idx]
        if len(idxs) >= 2:
            xs = x[idxs]
            ys = y[idxs]
            total += (xs.max() - xs.min()) + (ys.max() - ys.min())
    return total

#AD-> Calcultes the total cost along with area, wirelength,x,y,widths, and heights of blocks
def cost(sp_pos, sp_neg, blocks, block_names, nets, name_to_idx, alpha, beta,
         ref_area=None, ref_wl=None, verify=False):
    x, y, widths, heights = sp_to_placement(sp_pos, sp_neg, blocks, block_names)
    if verify:
        check_no_overlaps(x, y, widths, heights, block_names)
    W, H = compute_bounding_box(x, y, widths, heights)
    area = W * H
    wl   = compute_hpwl(x, y, nets, name_to_idx)
    a_norm = area / ref_area if ref_area else area
    w_norm = wl   / ref_wl   if ref_wl   else wl
    return alpha * a_norm + beta * w_norm, area, wl, x, y, widths, heights


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation moves
# ─────────────────────────────────────────────────────────────────────────────
#AD-> Swapes on a random basis
def perturb(sp_pos, sp_neg, blocks, block_names):
    """
    OPTIMIZATION: only copy the data structure that actually changes.
    Original copied all three every call regardless of move type.
    """
    n = len(sp_pos)
    move = random.randint(0, 2)

    if move == 0:
        # Swap two blocks in Γ+  — only sp_pos changes
        sp_pos = list(sp_pos)
        i, j = random.sample(range(n), 2)
        sp_pos[i], sp_pos[j] = sp_pos[j], sp_pos[i]
        return sp_pos, sp_neg, blocks

    elif move == 1:
        # Swap two blocks in Γ-  — only sp_neg changes
        sp_neg = list(sp_neg)
        i, j = random.sample(range(n), 2)
        sp_neg[i], sp_neg[j] = sp_neg[j], sp_neg[i]
        return sp_pos, sp_neg, blocks

    else:
        # Rotate a random block (swap w, h)  — only blocks changes
        blocks = dict(blocks)
        idx = random.randint(0, n - 1)
        name = block_names[idx]
        w, h = blocks[name]
        blocks[name] = (h, w)
        return sp_pos, sp_neg, blocks


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────
#AD-> Calcultes the orignal states and modified states of floorplan
def simulated_annealing(blocks_orig, block_names, nets, name_to_idx,
                        alpha, beta, verbose=True):
    n = len(block_names)
    blocks = dict(blocks_orig)
    start_time = time.time()

    # Initial sequence pair: random permutation
    sp_pos = list(range(n))
    sp_neg = list(range(n))
    random.shuffle(sp_pos)
    random.shuffle(sp_neg)

    # ── SA parameters ──────────────────────────────────────────────────
    # Estimate initial T from a short random walk
    NUM_PROBE = max(50, 5 * n)
    costs_probe = []
    sp_p_tmp, sp_n_tmp, blk_tmp = sp_pos[:], sp_neg[:], dict(blocks)
    for _ in range(NUM_PROBE):
        sp_p2, sp_n2, blk2 = perturb(sp_p_tmp, sp_n_tmp, blk_tmp, block_names)
        x, y, ws, hs = sp_to_placement(sp_p2, sp_n2, blk2, block_names)
        W, H = compute_bounding_box(x, y, ws, hs)
        costs_probe.append(W * H)
        sp_p_tmp, sp_n_tmp, blk_tmp = sp_p2, sp_n2, blk2
    delta_avg = max(abs(costs_probe[i] - costs_probe[i - 1])
                    for i in range(1, len(costs_probe)))
    T0 = delta_avg / math.log(2)   # accept ~50% bad moves initially

    # OPTIMIZATION: linear moves-per-temp instead of quadratic (10*n² → 20*n)
    # Slightly slower cooling (0.92 → 0.95) compensates in solution quality.
    MOVES_PER_TEMP = max(20 * n, 200)
    COOL_FACTOR    = 0.95
    T_MIN          = T0 * 1e-4
    MAX_ITERATIONS = int(math.log(T_MIN / T0) / math.log(COOL_FACTOR)) + 1

    # ── Collect reference costs ────────────────────────────────────────
    x0, y0, w0, h0 = sp_to_placement(sp_pos, sp_neg, blocks, block_names)
    W0, H0 = compute_bounding_box(x0, y0, w0, h0)
    ref_area = W0 * H0 if W0 * H0 > 0 else 1.0
    ref_wl   = compute_hpwl(x0, y0, nets, name_to_idx)
    if ref_wl == 0:
        ref_wl = 1.0

    cur_cost, cur_area, cur_wl, x, y, ws, hs = cost(
        sp_pos, sp_neg, blocks, block_names, nets, name_to_idx,
        alpha, beta, ref_area, ref_wl)

    best_cost = cur_cost
    best_sp_pos, best_sp_neg = sp_pos[:], sp_neg[:]
    best_blocks = dict(blocks)
    best_x, best_y, best_ws, best_hs = x.copy(), y.copy(), ws.copy(), hs.copy()
    best_area, best_wl = cur_area, cur_wl

    # Record initial state
    init_sp_pos, init_sp_neg = sp_pos[:], sp_neg[:]
    init_blocks = dict(blocks)
    init_x, init_y, init_ws, init_hs = x.copy(), y.copy(), ws.copy(), hs.copy()
    init_W, init_H = W0, H0
    init_area = cur_area
    init_wl   = cur_wl

    T = T0
    total_moves = 0
    accepted = 0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Simulated Annealing Parameters")
        print(f"{'─'*60}")
        print(f"  Blocks             : {n}")
        print(f"  Nets               : {len(nets)}")
        print(f"  Alpha (area)       : {alpha}")
        print(f"  Beta  (wirelength) : {beta}")
        print(f"  T0                 : {T0:.4f}")
        print(f"  T_min              : {T_MIN:.6f}")
        print(f"  Cooling factor     : {COOL_FACTOR}")
        print(f"  Moves/temperature  : {MOVES_PER_TEMP}")
        print(f"  Max temp steps     : {MAX_ITERATIONS}")
        print(f"  Total max moves    : {MAX_ITERATIONS * MOVES_PER_TEMP:,}")
        print(f"{'─'*60}\n")

    for step in range(MAX_ITERATIONS):
        T = T0 * (COOL_FACTOR ** step)
        if T < T_MIN:
            break

        for _ in range(MOVES_PER_TEMP):
            sp_pos2, sp_neg2, blocks2 = perturb(sp_pos, sp_neg, blocks, block_names)
            new_cost, new_area, new_wl, nx, ny, nws, nhs = cost(
                sp_pos2, sp_neg2, blocks2, block_names, nets, name_to_idx,
                alpha, beta, ref_area, ref_wl)

            delta = new_cost - cur_cost
            total_moves += 1

            if delta < 0 or random.random() < math.exp(-delta / T):
                sp_pos, sp_neg, blocks = sp_pos2, sp_neg2, blocks2
                cur_cost, cur_area, cur_wl = new_cost, new_area, new_wl
                x, y, ws, hs = nx, ny, nws, nhs
                accepted += 1

                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_sp_pos, best_sp_neg = sp_pos[:], sp_neg[:]
                    best_blocks = dict(blocks)
                    best_x, best_y, best_ws, best_hs = x.copy(), y.copy(), ws.copy(), hs.copy()
                    best_area, best_wl = cur_area, cur_wl

        if verbose and step % max(1, MAX_ITERATIONS // 10) == 0:
            elapsed = time.time() - start_time
            print(f"  T={T:10.4f}  cost={best_cost:.6f}  "
                  f"area={best_area:.1f}  wl={best_wl:.1f}  ")

    elapsed = time.time() - start_time
    accept_rate = accepted / total_moves * 100 if total_moves else 0

    best_W, best_H = compute_bounding_box(best_x, best_y, best_ws, best_hs)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  SA Results")
        print(f"{'─'*60}")
        print(f"  Total moves        : {total_moves:,}")
        print(f"  Accepted moves     : {accepted:,}")
        print(f"  Acceptance rate    : {accept_rate:.2f}%")
        print(f"  Total runtime      : {elapsed:.3f}s")
        print(f"{'─'*60}")
        print(f"  Initial area       : {init_area:.1f}")
        print(f"  Final   area       : {best_area:.1f}  ({100*(best_area-init_area)/init_area:+.1f}%)")
        print(f"  Initial HPWL       : {init_wl:.1f}")
        print(f"  Final   HPWL       : {best_wl:.1f}  ({100*(best_wl-init_wl)/max(init_wl,1):+.1f}%)")
        print(f"  Final bounding box : {best_W:.1f} x {best_H:.1f}")
        print(f"{'─'*60}\n")

        init_names_pos = [block_names[i] for i in init_sp_pos]
        init_names_neg = [block_names[i] for i in init_sp_neg]
        best_names_pos = [block_names[i] for i in best_sp_pos]
        best_names_neg = [block_names[i] for i in best_sp_neg]
        print(f"  Initial Γ+ : {init_names_pos}")
        print(f"  Initial Γ- : {init_names_neg}")
        print(f"  Final   Γ+ : {best_names_pos}")
        print(f"  Final   Γ- : {best_names_neg}")
        print()

    # Verify no overlaps in initial and final placements
    check_no_overlaps(init_x, init_y, init_ws, init_hs, block_names)
    check_no_overlaps(best_x, best_y, best_ws, best_hs, block_names)
    if verbose:
        print("  [OK] No overlaps in initial or final placement.")

    return (init_x, init_y, init_ws, init_hs, init_W, init_H, init_area, init_wl,
            init_sp_pos, init_sp_neg, init_blocks,
            best_x, best_y, best_ws, best_hs, best_W, best_H, best_area, best_wl,
            best_sp_pos, best_sp_neg, best_blocks,
            elapsed, total_moves, accepted)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

COLORS = [
    '#AED6F1','#A9DFBF','#F9E79F','#F1948A','#C39BD3',
    '#85C1E9','#82E0AA','#F7DC6F','#EC7063','#AF7AC5',
    '#76D7C4','#F0B27A','#7FB3D3','#7DCEA0','#F1948A',
]

#AD- Visualization of the floorplan
def draw_floorplan(ax, x, y, ws, hs, block_names, nets, name_to_idx,
                   bbox_W, bbox_H, title, global_max_x, global_max_y):
    ax.set_xlim(-1, global_max_x + 1)
    ax.set_ylim(-1, global_max_y + 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)

    # Bounding rectangle
    rect = patches.Rectangle((0, 0), bbox_W, bbox_H,
                               linewidth=2, edgecolor='black',
                               facecolor='none', linestyle='--', zorder=3)
    ax.add_patch(rect)

    # Draw blocks
    for i, name in enumerate(block_names):
        c = COLORS[i % len(COLORS)]
        r = patches.Rectangle((x[i], y[i]), ws[i], hs[i],
                               linewidth=1.2, edgecolor='#2C3E50',
                               facecolor=c, alpha=0.75, zorder=2)
        ax.add_patch(r)
        cx = x[i] + ws[i] / 2
        cy = y[i] + hs[i] / 2
        ax.text(cx, cy, name, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=4)

    # Draw net edges (lower-left corners)
    for net in nets:
        pts = []
        for bname in net:
            if bname in name_to_idx:
                idx = name_to_idx[bname]
                pts.append((x[idx], y[idx]))
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                ax.plot([pts[a][0], pts[b][0]],
                        [pts[a][1], pts[b][1]],
                        color='#E74C3C', linewidth=0.7,
                        alpha=0.55, zorder=1)

    ax.text(0.01, 0.99,
            f"BBox: {bbox_W:.1f}×{bbox_H:.1f}  Area: {bbox_W*bbox_H:.1f}",
            transform=ax.transAxes, fontsize=8.5,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

#AD-> Creates the image file to save
def save_comparison_png(result, block_names, nets, name_to_idx, output_path):
    (ix, iy, iws, ihs, iW, iH, iarea, iwl,
     isp_pos, isp_neg, iblocks,
     fx, fy, fws, fhs, fW, fH, farea, fwl,
     fsp_pos, fsp_neg, fblocks,
     elapsed, total_moves, accepted) = result

    global_max_x = max(iW, fW) * 1.05
    global_max_y = max(iH, fH) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    draw_floorplan(axes[0], ix, iy, iws, ihs, block_names, nets, name_to_idx,
                   iW, iH,
                   f"Initial  |  Area={iarea:.1f}  HPWL={iwl:.1f}",
                   global_max_x, global_max_y)

    draw_floorplan(axes[1], fx, fy, fws, fhs, block_names, nets, name_to_idx,
                   fW, fH,
                   f"Final  |  Area={farea:.1f}  HPWL={fwl:.1f}",
                   global_max_x, global_max_y)

    fig.text(0.5, 0.01,
             f"Moves: {total_moves:,}  |  Accepted: {accepted/total_moves*100:.1f}%",
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Floorplan image saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sequence-Pair Floorplanner with Simulated Annealing")
    parser.add_argument('--input',  required=True, help='Input design file')
    parser.add_argument('--alpha',  type=float, default=0.5,
                        help='Weight for bounding-box area  (default=0.5)')
    parser.add_argument('--beta',   type=float, default=0.5,
                        help='Weight for HPWL wirelength   (default=0.5)')
    parser.add_argument('--output', default='output.png',
                        help='Output PNG file (default=output.png)')
    parser.add_argument('--seed',   type=int, default=0,
                        help='Random seed (default=0)')
    args = parser.parse_args()

    if abs(args.alpha + args.beta - 1.0) > 1e-6:
        print(f"[WARN] alpha+beta = {args.alpha+args.beta:.4f} ≠ 1.0")

    random.seed(args.seed)

    blocks, nets = parse_input(args.input)
    block_names  = list(blocks.keys())
    name_to_idx  = {name: i for i, name in enumerate(block_names)}

    print(f"[INFO] Loaded {len(block_names)} blocks, {len(nets)} nets")

    result = simulated_annealing(
        blocks, block_names, nets, name_to_idx,
        args.alpha, args.beta, verbose=True)

    save_comparison_png(result, block_names, nets, name_to_idx, args.output)


if __name__ == '__main__':
    main()
