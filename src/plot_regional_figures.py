"""Same-scale before/after regional drawings; no fitted animal templates."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--before', required=True)
parser.add_argument('--after', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--style', choices=['simple', 'balanced', 'rich'], default='rich')
parser.add_argument('--regions', type=int, nargs='+', default=[0, 5, 10])
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
args = parser.parse_args()
if any(i < 0 or i >= 15 for i in args.regions) or any(i < 0 for i in args.seeds):
    parser.error('regions must be 0..14 and seeds must be nonnegative')
before, after = [json.loads(Path(p).read_text()) for p in (args.before, args.after)]
sky = after['sky']
lookup = {s['id']: s for s in sky['stars']}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for index in args.regions:
    c = sky['regions'][index]['center']
    lon, lat = np.radians([c['longitude'], c['latitude']])
    center = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.cross(center, east)

    def project(vectors):
        v = np.atleast_2d(vectors)
        factor = 2 / (1 + v @ center) * 180 / np.pi
        return np.c_[-(v @ east)*factor, (v @ north)*factor]

    pool = [s for s in sky['stars'] if s['region'] == index and s['app_mag'] <= 4.5]
    xy = project([s['direction'] for s in pool])
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) + 5
    middle = (xy.min(axis=0) + xy.max(axis=0)) / 2
    compact = len(args.seeds) == 1
    fig, axs = plt.subplots(2 if compact else 4, 2 if compact else len(args.seeds),
                            figsize=(9, 9) if compact else (3.25*len(args.seeds), 13), facecolor='#101924')
    for row, (report, core, label) in enumerate([(before, False, 'Before / full'), (after, False, 'After / full'),
                                               (before, True, 'Before / core'), (after, True, 'After / core')]):
        for col, seed in enumerate(args.seeds):
            ax = axs[row//2, row % 2] if compact else axs[row, col]
            ax.set_facecolor('#101924')
            sample = next(r for r in report['rows'] if r['index'] == index and r['style'] == args.style and r['seed'] == f'morphology-{seed}')
            f = sample['figure']
            ids, edges = (f['coreMembers'], f['coreEdges']) if core else (f['members'], f['edges'])
            ax.scatter(xy[:, 0], xy[:, 1], s=3, color='#738195', alpha=.55)
            for a, b in edges:
                u, v = [np.array(lookup[k]['direction']) for k in (a, b)]
                angle = np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v)
                t = np.linspace(0, 1, max(3, int(np.degrees(angle)*4)))
                curve = (np.sin((1-t)*angle)[:, None]*u + np.sin(t*angle)[:, None]*v) / np.sin(angle)
                p = project(curve)
                ax.plot(p[:, 0], p[:, 1], color='#63d4ce' if row % 2 else '#aab5c7', lw=1.15)
            for id in ids:
                s = lookup[id]
                x, y = project(s['direction'])[0]
                size = max(7, 9+(4.5-s['app_mag'])*11)
                ax.scatter(x, y, s=size, color=s['color_hex'], linewidth=0, zorder=3)
                if id in sample['required']:
                    ax.scatter(x, y, s=size+30, facecolors='none', edgecolors='#e8bb68', linewidth=.65, zorder=4)
            m = sample['core' if core else 'full']
            ax.set_title(f'{label}  |  seed {seed}\n{len(ids)} stars / {m["loops"]} cycles / {m["branches"]} branches', color='#dce6f2', fontsize=9)
            ax.set_xlim(middle[0]-span/2, middle[0]+span/2)
            ax.set_ylim(middle[1]-span/2, middle[1]+span/2)
            ax.set_aspect('equal')
            ax.axis('off')
    fig.suptitle(f'Z{index+1:02d}: same sky, same boundaries, same angular scale', color='white', fontsize=13 if compact else 16, y=.99)
    note = after.get('selectionNote', 'Fixed seeds: '+', '.join(map(str, args.seeds))+'. East left / north up (day-zero ecliptic).')
    fig.text(.5, .02, 'Gold rings: all required bright stars.\n'+note, color='#9fadc0', ha='center', fontsize=9)
    fig.subplots_adjust(left=.02, right=.98, top=.90 if compact else .94, bottom=.065, wspace=.08, hspace=.20)
    fig.savefig(out / f'Z{index+1:02d}.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
