from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Optional, List, Dict, Tuple

try:
    from textwrap import fill
    import igraph as ig
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path
except ImportError:
    fill = None
    ig = None
    mpl = None

import numpy as np
import rustworkx as rx

from typing_extensions import Any


@dataclass
class ColorLegend:
    name: str = field(default="Other")
    color: str = field(default="white")


# ---- rustworkx-backed node wrapper to mimic needed anytree.Node API ----
@dataclass
class RWXNode:
    name: str
    weight: str = field(default='')
    data: Optional[Any] = field(default=None)
    _primary_parent_id: Optional[int] = None
    color: ColorLegend = field(default_factory=ColorLegend)
    # Grouping/boxing options
    wrap_subtree: bool = field(default=False)
    wrap_facecolor: Optional[str] = field(default=None)
    wrap_edgecolor: Optional[str] = field(default=None)
    wrap_alpha: float = field(default=0.08)
    id_: int = field(init=False)
    _graph: ClassVar[rx.PyDAG] = rx.PyDAG()

    def __post_init__(self):
        # store self as node data to keep a 1:1 mapping
        self.id: int = self._graph.add_node(self)

    # Non-primary connect: add edge without changing primary parent pointer
    def add_parent(self, parent: "RWXNode", edge_weight=None):
        # Avoid self-loops
        if parent is self:
            return
        # Do not add duplicate edges between the same two nodes
        if self._graph.has_edge(parent.id, self.id):
            return
        # Avoid creating cycles: PyDAG will raise if creates a cycle
        self._graph.add_edge(parent.id, self.id, edge_weight if edge_weight is not None else self.weight)

    def remove(self):
        self._graph.remove_node(self.id)

    def remove_node(self, node: RWXNode):
        self._graph.remove_node(node.id)

    def remove_child(self, child: RWXNode):
        child.remove_parent(self)

    def remove_parent(self, parent: RWXNode):
        self._graph.remove_edge(parent.id, self.id)

    @property
    def ancestors(self) -> List[RWXNode]:
        node_ids = rx.ancestors(self._graph, self.id)
        return [self._graph[n_id] for n_id in node_ids]

    @property
    def parents(self)-> List[RWXNode]:
        # In this environment rustworkx returns node data objects directly
        return self._graph.predecessors(self.id)

    @property
    def parent(self) -> Optional["RWXNode"]:
        if self._primary_parent_id is None:
            return None
        return self._graph[self._primary_parent_id]

    @parent.setter
    def parent(self, value: Optional["RWXNode"]):
        if value is None:
            # detach current parent
            self._graph.remove_edge(self._primary_parent_id, self.id)
            self._primary_parent_id = None
            return
        # Create edge and set as primary (no need to detach non-primary edges)
        self.add_parent(value)
        self._primary_parent_id = value.id

    @property
    def children(self) -> List["RWXNode"]:
        # In this environment rustworkx returns node data objects directly
        return self._graph.successors(self.id)

    @property
    def descendants(self) -> List["RWXNode"]:
        desc_ids = rx.descendants(self._graph, self.id)
        return [self._graph[nid] for nid in desc_ids]

    @property
    def leaves(self) -> List["RWXNode"]:
        return [n for n in [self] + self.descendants if self._graph.out_degree(n.id) == 0]

    @property
    def root(self) -> "RWXNode":
        n = self
        while n.parent is not None:
            n = n.parent
        return n

    def __str__(self):
        return self.name

    def visualize(self, figsize=(30, 20), node_size=2000, font_size=10, spacing_x: float = 4, spacing_y: float = 4,
                  curve_scale: float = 0.5, layout: str = 'tidy', edge_style: str = 'spline',
                  label_max_chars_per_line: Optional[int] = 16):
        """Render a rooted, top-to-bottom directed graph (DAG-like), emphasizing
        flow from the root at the top to leaves at the bottom. Nodes are spread
        per layer with barycentric ordering to reduce crossings, and node sizes
        are adapted to fit their labels.

        Parameters:
        - layout: 'layered' (default) for the current layered layout, or 'tidy' for a tree-like tidy layout
                  based on each node's primary parent. Other values fall back to 'layered'.
        - edge_style: 'spline' (default) for Bezier spline routing, 'arc' for circular arcs,
                      'straight' for straight segments, or 'orthogonal' for L-shaped right-angled polylines.
        - label_max_chars_per_line: cap for wrapping long node labels; smaller values force more lines. Use None to disable the cap.
        """
        if not ig or not mpl or not fill:
            raise RuntimeError("igraph, matplotlib, textwrap must be installed to visualize the graph.")
        # 1) Build the rooted subgraph starting from the logical root
        root = self.root
        sub_nodes = [root] + root.descendants
        sub_id_set = {n.id for n in sub_nodes}

        # Stable deterministic node list (will be re-ordered by layers later)
        id_to_node = {n.id: n for n in sub_nodes}

        # Edges restricted to the rooted component
        rx_edges_all = list(self._graph.edge_list())
        rx_edges = [(u, v) for (u, v) in rx_edges_all if u in sub_id_set and v in sub_id_set]

        # Predecessor and successor maps (within the rooted subgraph)
        preds: Dict[int, List[int]] = defaultdict(list)
        succs: Dict[int, List[int]] = defaultdict(list)
        for (u, v) in rx_edges:
            preds[v].append(u)
            succs[u].append(v)

        # Ensure all nodes have entries in the maps
        for nid in sub_id_set:
            preds.setdefault(nid, [])
            succs.setdefault(nid, [])

        # 2) Longest-path layering from the root so all edges point downward
        layer: Dict[int, int] = {root.id: 0}
        from collections import deque
        dq = deque([root.id])
        while dq:
            u = dq.popleft()
            lu = layer.get(u, 0)
            for v in succs[u]:
                new_l = lu + 1
                if layer.get(v, -1) < new_l:
                    layer[v] = new_l
                    dq.append(v)
        # Some nodes may not have been reached (shouldn't happen for rooted subgraph), default to 0
        for nid in sub_id_set:
            layer.setdefault(nid, 0)

        max_layer = max(layer.values()) if layer else 0

        # Group nodes by layer
        layers: List[List[int]] = [[] for _ in range(max_layer + 1)] if max_layer >= 0 else []
        for nid, l in layer.items():
            layers[l].append(nid)
        # Initial within-layer order by node id for determinism
        for l in range(len(layers)):
            layers[l].sort()

        # 3) Barycentric ordering sweeps to reduce crossings
        def compute_order_index(layer_nodes: List[int]) -> Dict[int, int]:
            return {nid: idx for idx, nid in enumerate(layer_nodes)}

        def sort_by_barycenter(current_layer: List[int], reference_layer: List[int], use_preds: bool) -> List[int]:
            ref_pos = compute_order_index(reference_layer)
            def bary(nid: int) -> float:
                neighbors = preds[nid] if use_preds else succs[nid]
                neighbors = [w for w in neighbors if w in ref_pos]
                if not neighbors:
                    return float(ref_pos.get(nid, 0))  # keep stable
                return float(np.mean([ref_pos[w] for w in neighbors]))
            # Stable sort by (barycenter, id)
            return sorted(current_layer, key=lambda nid: (bary(nid), nid))

        # Perform a few downward/upward passes
        for _ in range(3):
            # Downward: order each layer based on predecessors in the layer above
            for l in range(1, len(layers)):
                layers[l] = sort_by_barycenter(layers[l], layers[l - 1], use_preds=True)
            # Upward: order each layer based on successors in the layer below
            for l in range(len(layers) - 2, -1, -1):
                layers[l] = sort_by_barycenter(layers[l], layers[l + 1], use_preds=False)

        # 4) Assign coordinates. Y by layer (top->bottom), X evenly within layer
        # Normalize to [0,1] with margins
        margin_x = 0.08
        margin_y = 0.08
        usable_w = 1.0 - 2 * margin_x
        usable_h = 1.0 - 2 * margin_y

        depth = max_layer + 1 if max_layer >= 0 else 1
        max_width = max((len(L) for L in layers), default=1)

        # Build final ordered node list and positions
        ordered_nodes: List[RWXNode] = []
        coords: List[Tuple[float, float]] = []

        if layout == 'tidy':
            # Build primary-child adjacency (tree) using each node's primary parent
            primary_children: Dict[int, List[int]] = defaultdict(list)
            for n in sub_nodes:
                pid = n._primary_parent_id
                if pid is not None and pid in sub_id_set:
                    primary_children[pid].append(n.id)
            # Ensure deterministic order of children
            for pid, ch in primary_children.items():
                ch.sort()

            # Assign x positions with a simple tidy-tree pass: leaves get consecutive
            # x values; internal nodes are centered above their children
            x_pos: Dict[int, float] = {}
            seen: set[int] = set()
            x_cursor = 0.0

            def assign_x(nid: int):
                nonlocal x_cursor
                children = primary_children.get(nid, [])
                if not children:
                    x_pos[nid] = x_cursor
                    x_cursor += 1.0
                else:
                    for c in children:
                        assign_x(c)
                    x_pos[nid] = float(np.mean([x_pos[c] for c in children]))
                seen.add(nid)

            # Start from the logical root; fall back to any unvisited nodes
            assign_x(root.id)
            # Handle nodes that are not connected via primary-parent to root
            for n in sorted(sub_nodes, key=lambda t: (layer[t.id], t.id)):
                if n.id not in seen:
                    assign_x(n.id)

            # Normalize x positions to [margin_x, 1 - margin_x]
            xs = list(x_pos.values())
            xmin = min(xs) if xs else 0.0
            xmax = max(xs) if xs else 1.0
            span = (xmax - xmin) if (xmax - xmin) > 1e-9 else 1.0

            x_norm: Dict[int, float] = {}
            for nid, xv in x_pos.items():
                x_norm[nid] = margin_x + ((xv - xmin) / span) * usable_w

            # Order nodes within each layer by their normalized x
            for l, layer_nodes in enumerate(layers):
                if not layer_nodes:
                    continue
                # y at top=1 down to bottom=0
                y_raw = 1.0 if depth == 1 else 1.0 - (l / float(max(depth - 1, 1)))
                y = margin_y + y_raw * usable_h
                # sort by x
                layer_nodes_sorted = sorted(layer_nodes, key=lambda nid: (x_norm.get(nid, 0.0), nid))
                for nid in layer_nodes_sorted:
                    x = x_norm.get(nid, margin_x + 0.5 * usable_w)
                    ordered_nodes.append(id_to_node[nid])
                    coords.append((x, y))
        else:
            # Default: layered equal-spacing placement
            for l, layer_nodes in enumerate(layers):
                k = max(1, len(layer_nodes))
                # y at top=1 down to bottom=0
                y_raw = 1.0 if depth == 1 else 1.0 - (l / float(max(depth - 1, 1)))
                y = margin_y + y_raw * usable_h
                # Even horizontal spacing within layer
                for i, nid in enumerate(layer_nodes):
                    x_rel = (i + 1) / float(k + 1)
                    x = margin_x + x_rel * usable_w
                    ordered_nodes.append(id_to_node[nid])
                    coords.append((x, y))

        if not ordered_nodes:
            # Handle empty graph gracefully
            return plt.subplots(figsize=(figsize or (6, 4)))

        norm_pos = np.array(coords, dtype=float)
        # Scale positions to increase spacing in tidy layout by stretching distances from margins.
        # This expands the data coordinate extents so nodes are farther apart regardless of figure size.
        x_extent, y_extent = 1.0, 1.0
        if layout == 'tidy':
            sx_eff = float(spacing_x)
            sy_eff = float(spacing_y)
            # Stretch distances from inner margins
            norm_pos[:, 0] = margin_x + (norm_pos[:, 0] - margin_x) * sx_eff
            norm_pos[:, 1] = margin_y + (norm_pos[:, 1] - margin_y) * sy_eff
            # Compute total extents including fixed outer margins
            x_extent = 2.0 * margin_x + usable_w * sx_eff
            y_extent = 2.0 * margin_y + usable_h * sy_eff
        # Map node id -> index in ordered_nodes
        id_map: Dict[int, int] = {n.id: i for i, n in enumerate(ordered_nodes)}

        # Re-map edges to ordered node indices (directed)
        edges = [(id_map[u], id_map[v]) for (u, v) in rx_edges if u in id_map and v in id_map]

        # 5) Adaptive figure size if not provided (based on layer grid)
        # Scale width primarily by the widest layer, and height by the number of layers.
        # Do not cap the scale so large graphs can become large and scrollable in interactive backends.
        if figsize is None:
            base_w, base_h = 12.0, 9.0
            # Rough character-based estimate to account for longer labels increasing node footprint
            label_lengths = [len(getattr(n, 'name', '')) for n in ordered_nodes]
            avg_label_len = float(np.mean(label_lengths)) if label_lengths else 0.0
            label_factor_w = 1.0 + min(2.0, avg_label_len / 30.0)

            # Estimate congestion: count nodes passing close to edges
            def _dist_pt_seg_l(px, py, ax_, ay_, bx_, by_):
                vx, vy = bx_ - ax_, by_ - ay_
                wx, wy = px - ax_, py - ay_
                vv = vx * vx + vy * vy
                if vv <= 1e-12:
                    return float(np.hypot(wx, wy))
                t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
                cx, cy = ax_ + t * vx, ay_ + t * vy
                return float(np.hypot(px - cx, py - cy))

            near_hits = 0
            thr = 0.06
            for (u_idx, v_idx) in edges:
                ax_, ay_ = norm_pos[u_idx, 0], norm_pos[u_idx, 1]
                bx_, by_ = norm_pos[v_idx, 0], norm_pos[v_idx, 1]
                xmin, xmax = min(ax_, bx_), max(ax_, bx_)
                ymin, ymax = min(ay_, by_), max(ay_, by_)
                for k, (xn, yn) in enumerate(norm_pos):
                    if k == u_idx or k == v_idx:
                        continue
                    if (xmin <= xn <= xmax) and (ymin <= yn <= ymax):
                        if _dist_pt_seg_l(xn, yn, ax_, ay_, bx_, by_) < thr:
                            near_hits += 1
            cong_factor = 1.0 + min(1.5, near_hits / float(max(1, len(ordered_nodes)))) * 0.35

            w_scale = max(1.0, (max_width / 2.0) * label_factor_w) * float(spacing_x) * cong_factor
            h_scale = max(1.0, (depth / 3.0)) * float(spacing_y) * cong_factor
            figsize = (base_w * w_scale, base_h * h_scale)
        fig, ax = plt.subplots(figsize=figsize)

        # Set axes limits early so pixel-to-data conversion reflects expanded extents
        ax.set_xlim(0.0, x_extent)
        ax.set_ylim(0.0, y_extent)

        # 6) Compute per-node label wrapping and size to fit text
        # Wrap width depends on the computed width scale to reduce node overlap
        wrap_width = 24
        labels = [n.name for n in ordered_nodes]
        if figsize is not None:
            # Start from a dynamic estimate based on figure width
            dynamic_width = max(16, int(24 * (figsize[0] / 12.0)))
            wrap_width = dynamic_width

        # Cap wrap width if requested to encourage multi-line labels
        if label_max_chars_per_line is not None:
            wrap_width = int(label_max_chars_per_line)

        wrapped_labels = [fill(lbl, width=wrap_width, break_long_words=True, break_on_hyphens=True) for lbl in labels]
        line_counts = [wl.count("\n") + 1 for wl in wrapped_labels]

        # Base node size can be adjusted for large graphs (n = total nodes)
        n_total = len(ordered_nodes)
        if n_total > 18:
            base_size = float(node_size) * max(0.72, min(1.0, float(np.sqrt(18.0 / n_total))))
        else:
            base_size = float(node_size)

        size_per_node = []
        for lines in line_counts:
            # Increase area with number of text lines
            scale = 1.0 + 0.70 * (lines - 1)
            size_per_node.append(base_size * scale)
        size_per_node = np.array(size_per_node, dtype=float)
        # Marker radius in points for each node (scatter size is points^2)
        radii_pt = np.sqrt(size_per_node / np.pi)

        # 7) Draw edges with arrowheads; support arc or spline routing with node avoidance
        arrow_color = "#666666"

        # Compute data-per-pixel conversion for cropping path at node borders
        fig_w_px, fig_h_px = fig.get_size_inches()[0] * fig.dpi, fig.get_size_inches()[1] * fig.dpi
        bbox = ax.get_position()
        ax_w_px = max(1.0, bbox.width * fig_w_px)
        ax_h_px = max(1.0, bbox.height * fig_h_px)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        dx_per_px = (xlim[1] - xlim[0]) / ax_w_px
        dy_per_px = (ylim[1] - ylim[0]) / ax_h_px
        radii_px = radii_pt * (fig.dpi / 72.0)
        radii_data_x = radii_px * dx_per_px
        radii_data_y = radii_px * dy_per_px

        # Precompute inflated obstacle rectangles (expanded ellipses) for node avoidance
        obstacles_rects: List[Tuple[float, float, float, float]] = []  # (xmin, xmax, ymin, ymax)
        # Use a small pixel-based inflation to give breathing room around nodes
        inflate_px_x = 3.0
        inflate_px_y = 3.0
        inflate_dx = dx_per_px * inflate_px_x
        inflate_dy = dy_per_px * inflate_px_y
        for k, (xn, yn) in enumerate(norm_pos):
            rx_k = float(abs(radii_data_x[k])) + inflate_dx
            ry_k = float(abs(radii_data_y[k])) + inflate_dy
            obstacles_rects.append((xn - rx_k, xn + rx_k, yn - ry_k, yn + ry_k))
        # Step sizes for scanning candidate guide lines
        avg_rx = float(np.mean([abs(v) for v in radii_data_x])) if len(radii_data_x) else 0.01
        avg_ry = float(np.mean([abs(v) for v in radii_data_y])) if len(radii_data_y) else 0.01
        step_x = max(0.02, 2.5 * avg_rx)
        step_y = max(0.02, 2.5 * avg_ry)
        # Accumulate previously drawn orthogonal segments (orientation, x0, y0, x1, y1, u_prev, v_prev)
        prev_ortho_segments: List[Tuple[str, float, float, float, float, int, int]] = []
        # Registry of bump centers to avoid duplicate bumps; store as pixel-quantized coordinates
        bump_registry: set = set()  # entries like ('h' or 'v', qx, qy)

        # 7.5) Draw transparent wrapping boxes for nodes marked to wrap their subtree
        wrap_roots = [n for n in ordered_nodes if getattr(n, 'wrap_subtree', False)]
        if wrap_roots:
            # Additional padding for the box in pixels -> data units
            box_pad_px = 12.0
            pad_dx = dx_per_px * box_pad_px
            pad_dy = dy_per_px * box_pad_px
            for r in wrap_roots:
                # Collect indices of r and its descendants that are in this plotted subgraph
                ids_in_subtree = {r.id}
                def wrap_children(node):
                    for child in node.children:
                        if child.wrap_subtree:
                            continue
                        ids_in_subtree.add(child.id)
                        if child.children:
                            wrap_children(child)
                wrap_children(r)
                idxs = [id_map[nid] for nid in ids_in_subtree if nid in id_map]
                if not idxs:
                    continue
                # Use already-inflated obstacle rectangles for node footprint extents
                xs_min = min(obstacles_rects[k][0] for k in idxs) - pad_dx
                xs_max = max(obstacles_rects[k][1] for k in idxs) + pad_dx
                ys_min = min(obstacles_rects[k][2] for k in idxs) - pad_dy
                ys_max = max(obstacles_rects[k][3] for k in idxs) + pad_dy
                # Clamp to axes limits (still allow small overflow)
                xs_min = max(0.0, xs_min)
                ys_min = max(0.0, ys_min)
                xs_max = min(x_extent, xs_max)
                ys_max = min(y_extent, ys_max)
                width = max(1e-6, xs_max - xs_min)
                height = max(1e-6, ys_max - ys_min)
                fc = r.wrap_facecolor if r.wrap_facecolor else (r.color.color if r.color else '#cccccc')
                ec = r.wrap_edgecolor if r.wrap_edgecolor else (r.color.color if r.color else '#666666')
                alpha_box = float(getattr(r, 'wrap_alpha', 0.08))
                box = FancyBboxPatch((xs_min, ys_min), width, height,
                                     boxstyle='round,pad=0.01',
                                     linewidth=2.0,
                                     edgecolor=ec,
                                     facecolor=fc,
                                     alpha=alpha_box,
                                     zorder=0.5,
                                     clip_on=False)
                ax.add_patch(box)

        for idx, (u, v) in enumerate(edges):
            x0, y0 = norm_pos[u, 0], norm_pos[u, 1]
            x1, y1 = norm_pos[v, 0], norm_pos[v, 1]

            # Bounding box and near-node analysis
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)
            near_count = 0
            min_d = 1.0
            nearest = None

            def _dist_pt_seg(px, py, ax_, ay_, bx_, by_):
                vx, vy = bx_ - ax_, by_ - ay_
                wx, wy = px - ax_, py - ay_
                vv = vx * vx + vy * vy
                if vv <= 1e-12:
                    return float(np.hypot(wx, wy))
                t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
                cx, cy = ax_ + t * vx, ay_ + t * vy
                return float(np.hypot(px - cx, py - cy))

            for k, (xn, yn) in enumerate(norm_pos):
                if k == u or k == v:
                    continue
                if (xmin <= xn <= xmax) and (ymin <= yn <= ymax):
                    near_count += 1
                    d = _dist_pt_seg(xn, yn, x0, y0, x1, y1)
                    if d < min_d:
                        min_d = d
                        nearest = (xn, yn)

            if edge_style == 'arc':
                # Backward compatible arc3 style with enhanced curvature
                dx = abs(x1 - x0)
                base_rad = 0.22 if dx >= 0.08 else 0.32
                if near_count:
                    base_rad *= min(1.0 + 0.18 * near_count, 2.2)
                if min_d < 0.08:
                    base_rad *= 1.2 + min(1.2, (0.08 - min_d) / 0.04 * 0.6)
                base_rad *= float(curve_scale)
                sign = 1.0 if (idx % 2 == 0) else -1.0
                rad = sign * base_rad

                shrinkA = float(max(1.0, radii_pt[u] + 2.0))
                shrinkB = float(max(1.0, radii_pt[v] + 2.0))
                arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                        arrowstyle='-|>',
                                        mutation_scale=26,
                                        linewidth=2.0,
                                        color=arrow_color,
                                        alpha=0.9,
                                        shrinkA=shrinkA,
                                        shrinkB=shrinkB,
                                        connectionstyle=f'arc3,rad={rad}',
                                        zorder=4,
                                        clip_on=False)
                ax.add_patch(arrow)
            elif edge_style == 'straight':
                shrinkA = float(max(1.0, radii_pt[u] + 2.0))
                shrinkB = float(max(1.0, radii_pt[v] + 2.0))
                arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                        arrowstyle='-|>',
                                        mutation_scale=16,
                                        linewidth=1.0,
                                        color=arrow_color,
                                        alpha=0.9,
                                        shrinkA=shrinkA,
                                        shrinkB=shrinkB,
                                        zorder=4,
                                        clip_on=False)
                ax.add_patch(arrow)
            elif edge_style in ('orthogonal', 'ortho', 'right_angle'):
                # Obstacle-avoiding orthogonal (Manhattan) routing.
                # Try to choose a single-bend HV or VH polyline that does not intersect any node box.
                # If no clear guide line is found, fall back to a spline to route around obstacles.

                # Helpers for collision detection against inflated node rectangles
                def _seg_blocked(p_start: Tuple[float, float], p_end: Tuple[float, float], ignore: Tuple[int, int]) -> bool:
                    xA, yA = p_start
                    xB, yB = p_end
                    if abs(yA - yB) <= 1e-12:
                        # Horizontal segment
                        y = yA
                        xmin_s, xmax_s = (xA, xB) if xA <= xB else (xB, xA)
                        for kk, (rxmin, rxmax, rymin, rymax) in enumerate(obstacles_rects):
                            if kk in ignore:
                                continue
                            if rymin <= y <= rymax and not (xmax_s < rxmin or xmin_s > rxmax):
                                return True
                        return False
                    elif abs(xA - xB) <= 1e-12:
                        # Vertical segment
                        x = xA
                        ymin_s, ymax_s = (yA, yB) if yA <= yB else (yB, yA)
                        for kk, (rxmin, rxmax, rymin, rymax) in enumerate(obstacles_rects):
                            if kk in ignore:
                                continue
                            if rxmin <= x <= rxmax and not (ymax_s < rymin or ymin_s > rymax):
                                return True
                        return False
                    else:
                        # Non-axis-aligned (shouldn't occur here)
                        return False

                def _within_bounds_x(xp: float) -> bool:
                    # Use expanded extents computed for tidy layout instead of assuming [0,1]
                    return (margin_x + 1e-3) <= xp <= (x_extent - margin_x - 1e-3)

                def _within_bounds_y(yp: float) -> bool:
                    # Use expanded extents computed for tidy layout instead of assuming [0,1]
                    return (margin_y + 1e-3) <= yp <= (y_extent - margin_y - 1e-3)

                ignore = (u, v)
                dx = x1 - x0
                dy = y1 - y0
                # Prefer a final vertical approach from above for downward (tree-like) edges
                prefer_top_entry = (y0 > y1 + 1e-6)
                # Default heuristic would choose HV if horizontal distance dominates, but for trees
                # we want VH so the arrow enters the target from above.
                use_hv_primary = (abs(dx) >= abs(dy)) and (not prefer_top_entry)

                # Long edge heuristic: if the direct source→target distance is a large fraction of
                # the graph's overall extent, prefer drawing this edge as a spline instead of
                # orthogonal to avoid excessive long, rigid polylines.
                diag = float(np.hypot(x_extent, y_extent))
                too_long_edge = bool(diag > 1e-9 and float(np.hypot(dx, dy)) > 0.4 * diag)

                # Minimum visible segment length to avoid zero-length legs (in data units)
                min_hlen = max(2.0 * dx_per_px * 4.0, 1e-4)
                min_vlen = max(2.0 * dy_per_px * 4.0, 1e-4)

                # Cropping distances at node borders (closer to node to ensure visual connection)
                crop_x_u = 1.02 * float(abs(radii_data_x[u]))
                crop_x_v = 1.02 * float(abs(radii_data_x[v]))
                crop_y_u = 1.02 * float(abs(radii_data_y[u]))
                crop_y_v = 1.02 * float(abs(radii_data_y[v]))
                # Compensate for arrowhead length so tip meets the node border (in pixels -> data units)
                tip_comp_px = 3.0
                tip_comp_dx = dx_per_px * tip_comp_px
                tip_comp_dy = dy_per_px * tip_comp_px

                found_path = False
                chosen_path: List[Tuple[float, float]] = []

                def try_hv() -> Optional[List[Tuple[float, float]]]:
                    # Horizontal then vertical with candidate guide lines x = xp
                    mx = 0.5 * (x0 + x1)
                    # Build candidate x positions scanning outward
                    candidates = [mx]
                    # Alternate left/right around mx
                    for i in range(1, 9):
                        off = step_x * i
                        candidates.append(mx + off)
                        candidates.append(mx - off)
                    # Also consider lines closer to source/target to escape local congestion
                    candidates.extend([x0 + np.sign(dx or 1.0) * step_x * 1.5,
                                       x1 - np.sign(dx or 1.0) * step_x * 1.5])

                    # Larger per-edge nudge to keep parallel edges from merging (in data units)
                    channel_px = 10.0
                    channel_dx = dx_per_px * channel_px
                    # Deterministic nudge order based on edge index
                    base_order = [0, 1, -1, 2, -2]
                    base = idx % len(base_order)
                    ordered_nudges = base_order[base:] + base_order[:base]

                    for xp in candidates:
                        for j in ordered_nudges:
                            xp_n = xp + j * channel_dx
                            if not _within_bounds_x(xp_n):
                                continue
                            # Ensure first and last horizontal legs have reasonable length
                            sx_sign = 1.0 if (xp_n - x0) > 0 else -1.0
                            ex_sign = 1.0 if (x1 - xp_n) > 0 else -1.0
                            start = (x0 + sx_sign * crop_x_u, y0)
                            # reduce cropping by tip compensation so arrowhead meets node
                            end = (x1 - ex_sign * max(0.0, (crop_x_v - tip_comp_dx)), y1)
                            if abs(xp_n - start[0]) < min_hlen:
                                continue
                            if abs(end[0] - xp_n) < min_hlen:
                                continue
                            p1 = (xp_n, y0)
                            p2 = (xp_n, y1)
                            if _seg_blocked(start, p1, ignore):
                                continue
                            if _seg_blocked(p1, p2, ignore):
                                continue
                            if _seg_blocked(p2, end, ignore):
                                continue
                            return [start, p1, p2, end]
                    return None

                def try_vh() -> Optional[List[Tuple[float, float]]]:
                    # Vertical then horizontal with candidate guide lines y = yp
                    my = 0.5 * (y0 + y1)
                    candidates = [my]
                    for i in range(1, 9):
                        off = step_y * i
                        candidates.append(my + off)
                        candidates.append(my - off)
                    candidates.extend([y0 + np.sign(dy or 1.0) * step_y * 1.5,
                                       y1 - np.sign(dy or 1.0) * step_y * 1.5])

                    # Larger per-edge nudge to keep parallel edges from merging (in data units)
                    channel_px = 10.0
                    channel_dy = dy_per_px * channel_px
                    base_order = [0, 1, -1, 2, -2]
                    base = idx % len(base_order)
                    ordered_nudges = base_order[base:] + base_order[:base]

                    for yp in candidates:
                        for j in ordered_nudges:
                            yp_n = yp + j * channel_dy
                            if not _within_bounds_y(yp_n):
                                continue
                            # For tree-like downward edges, ensure we approach the target from above
                            if prefer_top_entry and (yp_n <= y1 + min_vlen):
                                continue
                            # Never exit a node from the top: if the first leg would go upward from the source,
                            # skip this candidate. Require the first vertical segment to go downward (yp_n < y0).
                            if yp_n >= (y0 - min_vlen):
                                continue
                            sy_sign = 1.0 if (yp_n - y0) > 0 else -1.0
                            ey_sign = 1.0 if (y1 - yp_n) > 0 else -1.0
                            start = (x0, y0 + sy_sign * crop_y_u)
                            end = (x1, y1 - ey_sign * max(0.0, (crop_y_v - tip_comp_dy)))
                            if abs(yp_n - start[1]) < min_vlen:
                                continue
                            if abs(end[1] - yp_n) < min_vlen:
                                continue
                            p1 = (x0, yp_n)
                            p2 = (x1, yp_n)
                            if _seg_blocked(start, p1, ignore):
                                continue
                            if _seg_blocked(p1, p2, ignore):
                                continue
                            if _seg_blocked(p2, end, ignore):
                                continue
                            return [start, p1, p2, end]
                    return None

                # Two-bend option to preserve bottom exit and top entry (VHV)
                def try_vhv() -> Optional[List[Tuple[float, float]]]:
                    # Choose candidate y guide lines first (prefer above target, below source)
                    my = 0.5 * (y0 + y1)
                    y_candidates = [my]
                    for i in range(1, 9):
                        off = step_y * i
                        y_candidates.append(my + off)
                        y_candidates.append(my - off)
                    y_candidates.extend([
                        y0 - step_y * 1.5,  # a bit below source
                        y1 + step_y * 1.5,  # a bit above target
                    ])

                    # Channel separation in Y and X (reuse settings)
                    channel_px = 10.0
                    channel_dy = dy_per_px * channel_px
                    channel_dx = dx_per_px * channel_px
                    base_order = [0, 1, -1, 2, -2]
                    base = idx % len(base_order)
                    ordered_nudges = base_order[base:] + base_order[:base]

                    # Candidate X positions for the middle vertical leg
                    mx = 0.5 * (x0 + x1)
                    x_candidates = [mx]
                    for i in range(1, 9):
                        off = step_x * i
                        x_candidates.append(mx + off)
                        x_candidates.append(mx - off)
                    x_candidates.extend([
                        x0 + np.sign((x1 - x0) or 1.0) * step_x * 1.5,
                        x1 - np.sign((x1 - x0) or 1.0) * step_x * 1.5,
                    ])

                    for yp in y_candidates:
                        for jy in ordered_nudges:
                            yp_n = yp + jy * channel_dy
                            if not _within_bounds_y(yp_n):
                                continue
                            # Ensure vertical exit is indeed downward for tree-like edges
                            if prefer_top_entry:
                                if not (y1 + min_vlen < yp_n < y0 - min_vlen):
                                    continue
                            else:
                                if abs(yp_n - y0) < min_vlen or abs(y1 - yp_n) < min_vlen:
                                    continue

                            sy_sign = 1.0 if (yp_n - y0) > 0 else -1.0
                            ey_sign = 1.0 if (y1 - yp_n) > 0 else -1.0
                            start = (x0, y0 + sy_sign * crop_y_u)
                            mid_y_left = (x0, yp_n)

                            for xp in x_candidates:
                                for jx in ordered_nudges:
                                    xp_n = xp + jx * channel_dx
                                    if not _within_bounds_x(xp_n):
                                        continue
                                    # Check reasonable horizontal spans
                                    if abs(xp_n - x0) < min_hlen:
                                        continue
                                    if abs(x1 - xp_n) < min_hlen:
                                        continue
                                    mid_y_right = (x1, yp_n)
                                    mid_x_top = (xp_n, yp_n)
                                    mid_x_bottom = (xp_n, y1)
                                    end = (x1, y1 - ey_sign * max(0.0, (crop_y_v - tip_comp_dy)))

                                    # Collision checks for each orthogonal segment
                                    if _seg_blocked(start, mid_y_left, ignore):
                                        continue
                                    if _seg_blocked(mid_y_left, mid_x_top, ignore):
                                        continue
                                    if _seg_blocked(mid_x_top, mid_x_bottom, ignore):
                                        continue
                                    if _seg_blocked(mid_x_bottom, end, ignore):
                                        continue
                                    return [start, mid_y_left, mid_x_top, mid_x_bottom, end]
                    return None

                # Try in preferred orientation first, then enhanced alternatives
                path_points = None
                # If the edge goes upward or is on the same level (y1 >= y0), never let it exit from the top:
                # force a horizontal-first (HV) route to avoid upward vertical exit from the source.
                upward_or_same = (y1 >= (y0 - min_vlen))
                if upward_or_same:
                    path_points = try_hv()
                else:
                    # Downward (tree-like) edges: strongly prefer vertical exit from source (downward only)
                    path_points = try_vh() or try_vhv() or try_hv()

                if path_points is not None:
                    # Convert this edge to a spline if it would cross multiple other edges or if it is too long.
                    def _count_true_crosses(points: List[Tuple[float, float]]) -> int:
                        guard_px = 10.0
                        guard_dx = guard_px * dx_per_px
                        guard_dy = guard_px * dy_per_px
                        # Current path's own orthogonal segments
                        curr_segments: List[Tuple[str, float, float, float, float]] = []
                        for ii in range(len(points) - 1):
                            a = points[ii]
                            b = points[ii + 1]
                            if abs(a[1] - b[1]) <= 1e-12 and abs(a[0] - b[0]) > 1e-12:
                                curr_segments.append(('h', a[0], a[1], b[0], b[1]))
                            elif abs(a[0] - b[0]) <= 1e-12 and abs(a[1] - b[1]) > 1e-12:
                                curr_segments.append(('v', a[0], a[1], b[0], b[1]))

                        def _has_curr_vertical_at_x_through_y(xv: float, y: float, tol: float = 1e-9) -> bool:
                            for (ori2, xa2, ya2, xb2, yb2) in curr_segments:
                                if ori2 != 'v':
                                    continue
                                if abs(xa2 - xv) <= tol:
                                    yvmin2, yvmax2 = (ya2, yb2) if ya2 <= yb2 else (yb2, ya2)
                                    if yvmin2 - tol <= y <= yvmax2 + tol and abs(yb2 - ya2) > tol:
                                        return True
                            return False

                        def _has_curr_horizontal_at_y_through_x(yh: float, x: float, tol: float = 1e-9) -> bool:
                            for (ori2, xa2, ya2, xb2, yb2) in curr_segments:
                                if ori2 != 'h':
                                    continue
                                if abs(ya2 - yh) <= tol:
                                    xvmin2, xvmax2 = (xa2, xb2) if xa2 <= xb2 else (xb2, xa2)
                                    if xvmin2 - tol <= x <= xvmax2 + tol and abs(xb2 - xa2) > tol:
                                        return True
                            return False

                        count = 0
                        for (ori, xa, ya, xb, yb) in curr_segments:
                            if ori == 'h':
                                y = ya
                                xmin = min(xa, xb)
                                xmax = max(xa, xb)
                                for (pori, pxa, pya, pxb, pyb, pu, pv) in prev_ortho_segments:
                                    if pori != 'v':
                                        continue
                                    xv = pxa
                                    yvmin, yvmax = (pya, pyb) if pya <= pyb else (pyb, pya)
                                    if (xmin + guard_dx) <= xv <= (xmax - guard_dx) and (yvmin + guard_dy) <= y <= (yvmax - guard_dy):
                                        if _has_curr_vertical_at_x_through_y(xv, y):
                                            continue
                                        # Skip near shared incident nodes to avoid bumps/splines at splits
                                        sep_px = 30.0
                                        sep_dx = dx_per_px * sep_px
                                        sep_dy = dy_per_px * sep_px
                                        xu_c, yu_c = norm_pos[u, 0], norm_pos[u, 1]
                                        if pu == u and (abs(xv - xu_c) <= max(sep_dx, guard_dx * 1.5)) and (abs(y - yu_c) <= max(sep_dy, guard_dy * 1.5)):
                                            continue
                                        xv_c, yv_c = norm_pos[v, 0], norm_pos[v, 1]
                                        if pv == v and (abs(xv - xv_c) <= max(sep_dx, guard_dx * 1.5)) and (abs(y - yv_c) <= max(sep_dy, guard_dy * 1.5)):
                                            continue
                                        count += 1
                            elif ori == 'v':
                                x = xa
                                ymin = min(ya, yb)
                                ymax = max(ya, yb)
                                for (pori, pxa, pya, pxb, pyb, pu, pv) in prev_ortho_segments:
                                    if pori != 'h':
                                        continue
                                    yh = pya
                                    xvmin, xvmax = (pxa, pxb) if pxa <= pxb else (pxb, pxa)
                                    if (ymin + guard_dy) <= yh <= (ymax - guard_dy) and (xvmin + guard_dx) <= x <= (xvmax - guard_dx):
                                        if _has_curr_horizontal_at_y_through_x(yh, x):
                                            continue
                                        sep_px = 30.0
                                        sep_dx = dx_per_px * sep_px
                                        sep_dy = dy_per_px * sep_px
                                        xu_c, yu_c = norm_pos[u, 0], norm_pos[u, 1]
                                        if pu == u and (abs(x - xu_c) <= max(sep_dx, guard_dx * 1.5)) and (abs(yh - yu_c) <= max(sep_dy, guard_dy * 1.5)):
                                            continue
                                        xv_c, yv_c = norm_pos[v, 0], norm_pos[v, 1]
                                        if pv == v and (abs(x - xv_c) <= max(sep_dx, guard_dx * 1.5)) and (abs(yh - yv_c) <= max(sep_dy, guard_dy * 1.5)):
                                            continue
                                        count += 1
                        return count

                    crossing_count = _count_true_crosses(path_points)
                    if too_long_edge or crossing_count >= 3:
                        # Draw a spline instead of orthogonal for this specific edge
                        dvec = np.array([x1 - x0, y1 - y0], dtype=float)
                        dist = float(np.hypot(dvec[0], dvec[1]))
                        if dist >= 1e-9:
                            uvec = dvec / dist
                            pvec = np.array([-uvec[1], uvec[0]])
                            sA = 1.15 * float(np.hypot(radii_data_x[u] * uvec[0], radii_data_y[u] * uvec[1]))
                            sB = 1.15 * float(np.hypot(radii_data_x[v] * uvec[0], radii_data_y[v] * uvec[1]))
                            start = np.array([x0, y0]) + uvec * sA
                            end = np.array([x1, y1]) - uvec * sB
                            d2 = end - start
                            L = float(np.hypot(d2[0], d2[1]))
                            if L >= 1e-9:
                                offset = 0.12 * L
                                if near_count:
                                    offset *= min(1.0 + 0.22 * near_count, 2.5)
                                if min_d < 0.08:
                                    offset *= 1.0 + min(1.2, (0.08 - min_d) / 0.08 * 1.2)
                                offset *= float(curve_scale)
                                sign = 1.0 if (idx % 2 == 0) else -1.0
                                if nearest is not None:
                                    vec_to_near = np.array(nearest) - start
                                    if float(np.dot(pvec, vec_to_near)) > 0:
                                        sign = -sign
                                c1 = start + d2 * (1.0 / 3.0) + pvec * (offset * sign)
                                c2 = start + d2 * (2.0 / 3.0) + pvec * (offset * sign)
                                path = Path([tuple(start), tuple(c1), tuple(c2), tuple(end)],
                                            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                                arrow = FancyArrowPatch(path=path,
                                                        arrowstyle='-|>',
                                                        mutation_scale=16,
                                                        linewidth=4.0,
                                                        color=arrow_color,
                                                        alpha=0.5,
                                                        zorder=4,
                                                        clip_on=False)
                                ax.add_patch(arrow)
                        continue

                    # Insert small "overpass" bumps where this polyline crosses previously drawn orthogonal segments
                    def _insert_bumps(points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[int]]:
                        verts_out: List[Tuple[float, float]] = []
                        codes_out: List[int] = []
                        # Bump geometry in data units derived from pixels
                        bump_len_px = 14.0
                        bump_ht_px = 9.0
                        guard_px = 10.0
                        half_dx = (bump_len_px * 0.5) * dx_per_px
                        half_dy = (bump_len_px * 0.5) * dy_per_px
                        bump_hx = bump_ht_px * dx_per_px
                        bump_hy = bump_ht_px * dy_per_px
                        guard_dx = guard_px * dx_per_px
                        guard_dy = guard_px * dy_per_px

                        def add_moveto(pt: Tuple[float, float]):
                            verts_out.append(pt)
                            codes_out.append(Path.MOVETO)

                        def add_lineto(pt: Tuple[float, float]):
                            verts_out.append(pt)
                            codes_out.append(Path.LINETO)

                        def add_cubic(c1: Tuple[float, float], c2: Tuple[float, float], end: Tuple[float, float]):
                            verts_out.extend([c1, c2, end])
                            codes_out.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

                        # Precompute this edge's own orthogonal segments to detect coincident merges
                        curr_segments: List[Tuple[str, float, float, float, float]] = []
                        for ii in range(len(points) - 1):
                            a = points[ii]
                            b = points[ii + 1]
                            if abs(a[1] - b[1]) <= 1e-12 and abs(a[0] - b[0]) > 1e-12:
                                curr_segments.append(('h', a[0], a[1], b[0], b[1]))
                            elif abs(a[0] - b[0]) <= 1e-12 and abs(a[1] - b[1]) > 1e-12:
                                curr_segments.append(('v', a[0], a[1], b[0], b[1]))

                        def _has_curr_vertical_at_x_through_y(xv: float, y: float, tol: float = 1e-9) -> bool:
                            for (ori2, xa2, ya2, xb2, yb2) in curr_segments:
                                if ori2 != 'v':
                                    continue
                                if abs(xa2 - xv) <= tol:
                                    yvmin2, yvmax2 = (ya2, yb2) if ya2 <= yb2 else (yb2, ya2)
                                    if yvmin2 - tol <= y <= yvmax2 + tol and abs(yb2 - ya2) > tol:
                                        return True
                            return False

                        def _has_curr_horizontal_at_y_through_x(yh: float, x: float, tol: float = 1e-9) -> bool:
                            for (ori2, xa2, ya2, xb2, yb2) in curr_segments:
                                if ori2 != 'h':
                                    continue
                                if abs(ya2 - yh) <= tol:
                                    xvmin2, xvmax2 = (xa2, xb2) if xa2 <= xb2 else (xb2, xa2)
                                    if xvmin2 - tol <= x <= xvmax2 + tol and abs(xb2 - xa2) > tol:
                                        return True
                            return False

                        add_moveto(points[0])
                        for i in range(len(points) - 1):
                            A = points[i]
                            B = points[i + 1]
                            xA, yA = A
                            xB, yB = B
                            if abs(yA - yB) <= 1e-12 and abs(xA - xB) > 1e-12:
                                # Horizontal segment
                                y = yA
                                xmin = min(xA, xB)
                                xmax = max(xA, xB)
                                inters: List[float] = []
                                for (ori, xa, ya, xb, yb, pu, pv) in prev_ortho_segments:
                                    if ori != 'v':
                                        continue
                                    xv = xa
                                    yvmin, yvmax = (ya, yb) if ya <= yb else (yb, ya)
                                    if (xmin + guard_dx) <= xv <= (xmax - guard_dx) and (yvmin + guard_dy) <= y <= (yvmax - guard_dy):
                                        # If this edge turns and runs vertically along this x, skip bump (coincident merge)
                                        if _has_curr_vertical_at_x_through_y(xv, y):
                                            continue
                                        # Skip bumps very near a shared incident node (edges splitting from same node)
                                        sep_px = 30.0
                                        sep_dx = dx_per_px * sep_px
                                        sep_dy = dy_per_px * sep_px
                                        # Shared source near source node center
                                        xu_c, yu_c = norm_pos[u, 0], norm_pos[u, 1]
                                        if pu == u and (abs(xv - xu_c) <= max(sep_dx, guard_dx * 1.5)) and (abs(y - yu_c) <= max(sep_dy, guard_dy * 1.5)):
                                            continue
                                        # Shared target near target node center
                                        xv_c, yv_c = norm_pos[v, 0], norm_pos[v, 1]
                                        if pv == v and (abs(xv - xv_c) <= max(sep_dx, guard_dx * 1.5)) and (abs(y - yv_c) <= max(sep_dy, guard_dy * 1.5)):
                                            continue
                                        inters.append(xv)
                                if inters:
                                    inters = sorted(inters, key=lambda v: v if xB >= xA else -v)
                                    cursor_x = xA
                                    # helper to quantize a point to pixel grid
                                    def _qxy(x, y):
                                        qx = int(round(x / max(1e-12, dx_per_px)))
                                        qy = int(round(y / max(1e-12, dy_per_px)))
                                        return qx, qy
                                    for xi in inters:
                                        if (xB >= xA and xi <= cursor_x + guard_dx) or (xB < xA and xi >= cursor_x - guard_dx):
                                            continue
                                        remaining = (xmax - cursor_x) if xB >= xA else (cursor_x - xmin)
                                        half = min(half_dx, 0.5 * max(guard_dx, remaining * 0.25))
                                        start_x = xi - half
                                        end_x = xi + half
                                        start_x = max(start_x, xmin + guard_dx)
                                        end_x = min(end_x, xmax - guard_dx)
                                        if end_x - start_x < max(guard_dx * 0.6, 1e-6):
                                            continue
                                        # Prefer horizontal bumps; skip if a bump already exists at this crossing
                                        q = _qxy(xi, y)
                                        if ('v', q[0], q[1]) in bump_registry or ('h', q[0], q[1]) in bump_registry:
                                            # do not create another bump here
                                            continue
                                        add_lineto((start_x, y))
                                        # Choose bump direction (up or down) based on available room
                                        midx = 0.5 * (start_x + end_x)
                                        up_y = y + bump_hy
                                        dn_y = y - bump_hy
                                        ok_up = _within_bounds_y(up_y) and not _seg_blocked((midx, y), (midx, up_y), (-1, -1))
                                        ok_dn = _within_bounds_y(dn_y) and not _seg_blocked((midx, y), (midx, dn_y), (-1, -1))
                                        sign = 1.0 if ok_up or not ok_dn else -1.0
                                        c1 = (start_x + (end_x - start_x) * 0.35, y + sign * bump_hy)
                                        c2 = (start_x + (end_x - start_x) * 0.65, y + sign * bump_hy)
                                        add_cubic(c1, c2, (end_x, y))
                                        # register this bump center using the intersection point
                                        bump_registry.add(('h', q[0], q[1]))
                                        cursor_x = end_x
                                    add_lineto((xB, y))
                                else:
                                    add_lineto(B)
                            elif abs(xA - xB) <= 1e-12 and abs(yA - yB) > 1e-12:
                                # Vertical segment - prefer horizontal bumps only; do not add vertical bumps.
                                add_lineto(B)
                            else:
                                # Non-orthogonal (shouldn't happen here); keep straight
                                add_lineto(B)
                        return verts_out, codes_out

                    final_verts, final_codes = _insert_bumps(path_points)
                    path = Path(final_verts, final_codes)
                    arrow = FancyArrowPatch(path=path,
                                            arrowstyle='-|>',
                                            mutation_scale=26,
                                            linewidth=2.2,
                                            color=arrow_color,
                                            alpha=0.9,
                                            zorder=4,
                                            clip_on=False)
                    ax.add_patch(arrow)
                    # Register this edge's straight segments for later edges
                    for i2 in range(len(path_points) - 1):
                        a = path_points[i2]
                        b = path_points[i2 + 1]
                        if abs(a[0] - b[0]) <= 1e-12 and abs(a[1] - b[1]) > 1e-12:
                            prev_ortho_segments.append(('v', a[0], a[1], b[0], b[1], u, v))
                        elif abs(a[1] - b[1]) <= 1e-12 and abs(a[0] - b[0]) > 1e-12:
                            prev_ortho_segments.append(('h', a[0], a[1], b[0], b[1], u, v))
                else:
                    # Fallback: draw a spline that better avoids nodes (kept as last resort)
                    dvec = np.array([x1 - x0, y1 - y0], dtype=float)
                    dist = float(np.hypot(dvec[0], dvec[1]))
                    if dist < 1e-9:
                        continue
                    uvec = dvec / dist
                    pvec = np.array([-uvec[1], uvec[0]])

                    sA = 1.15 * float(np.hypot(radii_data_x[u] * uvec[0], radii_data_y[u] * uvec[1]))
                    sB = 1.15 * float(np.hypot(radii_data_x[v] * uvec[0], radii_data_y[v] * uvec[1]))
                    start = np.array([x0, y0]) + uvec * sA
                    end = np.array([x1, y1]) - uvec * sB
                    d2 = end - start
                    L = float(np.hypot(d2[0], d2[1]))
                    if L < 1e-9:
                        continue

                    offset = 0.12 * L
                    if near_count:
                        offset *= min(1.0 + 0.22 * near_count, 2.5)
                    if min_d < 0.08:
                        offset *= 1.0 + min(1.2, (0.08 - min_d) / 0.08 * 1.2)
                    offset *= float(curve_scale)

                    sign = 1.0 if (idx % 2 == 0) else -1.0
                    if nearest is not None:
                        vec_to_near = np.array(nearest) - start
                        if float(np.dot(pvec, vec_to_near)) > 0:
                            sign = -sign

                    c1 = start + d2 * (1.0 / 3.0) + pvec * (offset * sign)
                    c2 = start + d2 * (2.0 / 3.0) + pvec * (offset * sign)
                    path = Path([tuple(start), tuple(c1), tuple(c2), tuple(end)],
                                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                    arrow = FancyArrowPatch(path=path,
                                            arrowstyle='-|>',
                                            mutation_scale=16,
                                            linewidth=4.0,
                                            color=arrow_color,
                                            alpha=0.5,
                                            zorder=4,
                                            clip_on=False)
                    ax.add_patch(arrow)
            else:
                # Spline routing: cubic Bezier with obstacle-aware lateral offset
                dvec = np.array([x1 - x0, y1 - y0], dtype=float)
                dist = float(np.hypot(dvec[0], dvec[1]))
                if dist < 1e-9:
                    continue
                uvec = dvec / dist
                pvec = np.array([-uvec[1], uvec[0]])

                # Crop endpoints to node borders (approximate ellipse per node)
                sA = 1.15 * float(np.hypot(radii_data_x[u] * uvec[0], radii_data_y[u] * uvec[1]))
                sB = 1.15 * float(np.hypot(radii_data_x[v] * uvec[0], radii_data_y[v] * uvec[1]))
                start = np.array([x0, y0]) + uvec * sA
                end = np.array([x1, y1]) - uvec * sB
                d2 = end - start
                L = float(np.hypot(d2[0], d2[1]))
                if L < 1e-9:
                    continue

                # Base lateral offset proportional to edge length and congestion
                offset = 0.12 * L
                if near_count:
                    offset *= min(1.0 + 0.22 * near_count, 2.5)
                if min_d < 0.08:
                    offset *= 1.0 + min(1.2, (0.08 - min_d) / 0.08 * 1.2)
                offset *= float(curve_scale)

                # Choose side to push away from nearest obstacle if available
                sign = 1.0 if (idx % 2 == 0) else -1.0
                if nearest is not None:
                    vec_to_near = np.array(nearest) - start
                    if float(np.dot(pvec, vec_to_near)) > 0:
                        sign = -sign

                c1 = start + d2 * (1.0 / 3.0) + pvec * (offset * sign)
                c2 = start + d2 * (2.0 / 3.0) + pvec * (offset * sign)

                path = Path([tuple(start), tuple(c1), tuple(c2), tuple(end)],
                            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])

                arrow = FancyArrowPatch(path=path,
                                        arrowstyle='-|>',
                                        mutation_scale=16,
                                        linewidth=4.0,
                                        color=arrow_color,
                                        alpha=0.5,
                                        zorder=4,
                                        clip_on=False)
                ax.add_patch(arrow)

        # 8) Draw nodes
        # Color nodes by expression class family. Defaults remain blue for others.
        colors = [n.color.color if n.color else ColorLegend().color for n in ordered_nodes]

        ax.scatter(norm_pos[:, 0], norm_pos[:, 1],
                   s=size_per_node, c=colors,
                   edgecolors='black', linewidths=2.0,
                   alpha=0.95, zorder=2)

        # 9) Draw labels centered in nodes
        for (x, y), text in zip(norm_pos, wrapped_labels):
            ax.text(x, y, text,
                    ha='center', va='center',
                    fontsize=font_size, fontweight='medium', wrap=True,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.85),
                    zorder=3)

        # 9.5) Add a legend for node colors (by expression family)
        from collections import OrderedDict
        fam_to_color = OrderedDict()
        for n, col in zip(ordered_nodes, colors):
            label = n.color.name if n.color else ColorLegend().name
            # Only include categories actually present
            if label not in fam_to_color:
                fam_to_color[label] = col
        # Build legend handles
        handles = []
        for lbl, col in fam_to_color.items():
            patch = mpl.patches.Patch(facecolor=col, edgecolor='black', label=lbl)
            handles.append(patch)
        if handles:
            # Place legend inside axes to avoid resizing; semi-transparent frame
            # Scale legend font size with figure size so it grows/shrinks with the visualization
            fw, fh = fig.get_size_inches()
            # scale relative to a 12x9 inch base figure, with guards for very small sizes
            scale = 0.5 * (max(0.5, fw / 12.0) + max(0.5, fh / 9.0))
            legend_fs = float(np.clip(10.0 * scale, 8.0, 28.0))
            title_fs = float(np.clip(legend_fs * 1.1, 9.0, 32.0))
            ax.legend(handles=handles, title="Node types", loc='upper left', framealpha=0.9,
                      fontsize=legend_fs, title_fontsize=title_fs)

        # 10) Axes styling and limits
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.25)
        ax.set_title("Directed Query Graph (Top to Bottom)", fontsize=14, pad=20)
        # Allow non-square axes so width can expand with large layers
        ax.set_aspect('auto', adjustable='box')
        # Keep the expanded extents computed earlier (x_extent/y_extent)
        ax.set_xlim(0.0, x_extent)
        ax.set_ylim(0.0, y_extent)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return fig, ax
