import itertools
import sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

marker_names = [m for m in lines[3].strip("\n").split("\t") if m][2:]
data_row = lines[5].rstrip("\n").split("\t")
frame, t = data_row[0], data_row[1]
coords = data_row[2:]

print(f"{len(marker_names)} markers, frame {frame} t={t}")
print()
positions = {}
for i, name in enumerate(marker_names):
    x, y, z = float(coords[i * 3]), float(coords[i * 3 + 1]), float(coords[i * 3 + 2])
    positions[name] = (x, y, z)
    print(f"{name:12s} {x:8.4f} {y:8.4f} {z:8.4f}")

xs = [p[0] for p in positions.values()]
ys = [p[1] for p in positions.values()]
zs = [p[2] for p in positions.values()]
print()
print("bounding box:")
print("X range:", max(xs) - min(xs))
print("Y range:", max(ys) - min(ys))
print("Z range:", max(zs) - min(zs))

dists = []
for a, b in itertools.combinations(positions.keys(), 2):
    ax, ay, az = positions[a]
    bx, by, bz = positions[b]
    d = ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5
    dists.append(d)
print("min pairwise dist:", min(dists))
print("max pairwise dist:", max(dists))
