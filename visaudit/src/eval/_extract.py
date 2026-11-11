"""Chart property extraction. Runs INSIDE the sandbox subprocess.

Shipped into the temp cwd next to the plotting script, so it may import nothing
from this project: matplotlib / plotly are the only dependencies. It turns a
live figure object into a plain nested dict ("spec") that survives JSON.
"""
import base64
import hashlib
import json
import math
import re
import unicodedata
import zlib


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 10)


def _hex(c):
    try:
        from matplotlib.colors import to_hex
        return to_hex(c, keep_alpha=False)
    except Exception:
        return None


def _rgba(c):
    try:
        from matplotlib.colors import to_rgba
        return [_num(value) for value in to_rgba(c)]
    except Exception:
        return None


def _color_visible(c):
    rgba = _rgba(c)
    return rgba is not None and rgba[3] is not None and rgba[3] > 0


def _text(t):
    try:
        if hasattr(t, "get_text") and (
            not _artist_visible(t) or not _color_visible(t.get_color())
        ):
            return None
        s = t.get_text() if hasattr(t, "get_text") else str(t)
    except Exception:
        return None
    s = "".join(
        " " if char.isspace() else ""
        if unicodedata.category(char).startswith("C") else char
        for char in (s or "")
    )
    s = " ".join(s.split())
    return s or None


def _label(artist, fallback):
    try:
        lab = artist.get_label()
    except Exception:
        lab = None
    if not lab or lab.startswith("_"):
        return fallback
    return lab


def _artist_visible(artist):
    try:
        from matplotlib.axes import Axes
        from matplotlib.axis import Axis
        from matplotlib.figure import FigureBase
        from matplotlib.legend import Legend

        if not artist.get_visible():
            return False
        alpha = artist.get_alpha()
        # Container alpha is not inherited by children in Matplotlib. Only
        # primitive artists' alpha controls whether their marks are drawn.
        if (not isinstance(artist, (Axes, Axis, FigureBase, Legend))
                and alpha is not None and float(alpha) <= 0):
            return False
        # Matplotlib leaves children's flags unchanged when hiding their axes
        # or figure. Those children are still present but never drawn.
        axis = getattr(artist, "axes", None)
        if (hasattr(axis, "get_visible") and axis is not artist
                and not _artist_visible(axis)):
            return False
        figure = artist.get_figure()
        return figure is None or figure is artist or _artist_visible(figure)
    except Exception:
        return False


def _legend_handle_signature(handle):
    """The displayed swatch, independent of its position and internal object id."""
    if handle is None or not _artist_visible(handle):
        return None
    signature = {"kind": type(handle).__name__}
    for field in (
        "color", "facecolor", "edgecolor", "linewidth", "linestyle", "hatch",
        "marker", "markersize", "markerfacecolor", "markeredgecolor",
        "markeredgewidth", "fillstyle", "alpha", "sizes",
    ):
        getter = getattr(handle, "get_" + field, None)
        if not callable(getter):
            continue
        value = getter()
        if "color" in field:
            rgba = _rgba(value)
            if rgba is not None:
                value = rgba
            elif value is not None and not isinstance(value, str):
                value = [_rgba(c) for c in value]
        encoded = _stable_policy_value(value)
        if encoded is not _POLICY_UNSUPPORTED:
            signature[field] = encoded
    return signature


def _legend_items(legend):
    if legend is None or not _artist_visible(legend):
        return []
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    texts = list(legend.get_texts())
    return [
        {"label": label, "text": text,
         "handle": handles[index] if index < len(handles) else None}
        for index, text in enumerate(texts)
        if (label := _text(text))
    ]


def _legend_entries(items):
    return [{"label": item["label"], "handle": _legend_handle_signature(item["handle"])}
            for item in items]


def _with_capture_artists(item, artists):
    item["_capture_artists"] = list(artists)
    return item


def _series_root(axis_index, series, index):
    raw_label = (series[index].get("label") or "s%d" % index).strip()
    twins = [
        position for position, item in enumerate(series)
        if (item.get("label") or "") == series[index].get("label")
    ]
    label = "".join(
        {"%": "%25", "#": "%23", "[": "%5B", "]": "%5D"}.get(char, char)
        for char in raw_label
    )
    if len(twins) > 1:
        label = "%s#%d" % (label, twins.index(index))
    return "axes[%d].series[%s]" % (axis_index, label)


def _canvas_rgb(fig):
    import numpy as np

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


_POLICY_UNSUPPORTED = object()


def _stable_policy_value(value, depth=0):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _num(value)
    if callable(value):
        return _POLICY_UNSUPPORTED
    if isinstance(value, dict):
        result = {}
        for key in sorted(value, key=str):
            encoded = _stable_policy_value(value[key], depth + 1)
            if encoded is not _POLICY_UNSUPPORTED:
                result[str(key)] = encoded
        return result
    if isinstance(value, (list, tuple)):
        encoded = [_stable_policy_value(item, depth + 1) for item in value]
        return [item for item in encoded if item is not _POLICY_UNSUPPORTED]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _stable_policy_value(tolist(), depth + 1)
        except Exception:
            return _POLICY_UNSUPPORTED
    if depth < 2 and hasattr(value, "__dict__"):
        result = {}
        for key, item in sorted(vars(value).items()):
            if key in ("axis", "_axis"):
                continue
            encoded = _stable_policy_value(item, depth + 1)
            if encoded is not _POLICY_UNSUPPORTED:
                result[key] = encoded
        return result
    return _POLICY_UNSUPPORTED


def _tick_component_policy(component, *, formatter):
    dynamic = {
        "axis", "_axis", "locs", "_locs", "offset", "_offset_string",
        "orderOfMagnitude", "_orderOfMagnitude", "format", "_format",
    } if formatter else {"axis", "_axis"}
    config = {}
    for key, value in sorted(vars(component).items()):
        if key in dynamic:
            continue
        encoded = _stable_policy_value(value)
        if encoded is not _POLICY_UNSUPPORTED:
            config[key] = encoded
    try:
        from matplotlib.ticker import FixedFormatter, FixedLocator
        fixed = isinstance(component, FixedFormatter if formatter else FixedLocator)
    except Exception:
        fixed = True
    return {
        "class": "%s.%s" % (type(component).__module__, type(component).__name__),
        "automatic": not fixed,
        "config": config,
    }


def _tick_policy(axis):
    return {
        "locator": _tick_component_policy(
            axis.get_major_locator(), formatter=False,
        ),
        "formatter": _tick_component_policy(
            axis.get_major_formatter(), formatter=True,
        ),
    }


def _heatmap_cell_display_position(artist, point_index):
    """Return the display-space center of one row-major heatmap cell."""
    import numpy as np

    array = np.ma.asarray(artist.get_array())
    coordinates_fn = getattr(artist, "get_coordinates", None)
    if callable(coordinates_fn):
        coordinates = np.asarray(coordinates_fn())
        if coordinates.ndim != 3 or coordinates.shape[2] != 2 or \
                coordinates.shape[0] < 2 or coordinates.shape[1] < 2:
            return None
        rows, columns = coordinates.shape[0] - 1, coordinates.shape[1] - 1
        if point_index >= rows * columns:
            return None
        row, column = divmod(point_index, columns)
        center = coordinates[row:row + 2, column:column + 2].mean(axis=(0, 1))
    else:
        if array.ndim < 2:
            return None
        rows, columns = array.shape[:2]
        if point_index >= rows * columns:
            return None
        row, column = divmod(point_index, columns)
        left, right, bottom, top = artist.get_extent()
        vertical_start, vertical_end = (
            (top, bottom) if getattr(artist, "origin", None) == "upper"
            else (bottom, top)
        )
        center = (
            left + (column + 0.5) * (right - left) / columns,
            vertical_start + (row + 0.5) *
            (vertical_end - vertical_start) / rows,
        )
    transformed = artist.get_transform().transform(center)
    return np.asarray(transformed, dtype=float)


def _aligned_heatmap_annotations(entry, point_index):
    """Find visible Text artists anchored at the selected heatmap cell."""
    import numpy as np

    artists = entry.get("artists") or []
    if len(artists) != 1:
        return []
    center = _heatmap_cell_display_position(artists[0], point_index)
    if center is None or center.shape != (2,) or not np.isfinite(center).all():
        return []
    aligned = []
    for text in entry.get("annotation_artists") or []:
        try:
            if not _artist_visible(text) or not (text.get_text() or "").strip():
                continue
            anchor = np.asarray(
                text.get_transform().transform(text.get_position()), dtype=float,
            )
            if anchor.shape == (2,) and np.isfinite(anchor).all() and \
                    np.linalg.norm(anchor - center) <= 3.0:
                aligned.append(text)
        except Exception:
            continue
    return aligned


def _point_mutation(entry, point_index):
    """Hide only one rendered point and return a restoration callback."""
    import numpy as np

    artists = entry.get("artists") or []
    kind = entry.get("kind")
    if not isinstance(point_index, int) or point_index < 0:
        return None
    if kind in ("bar", "barh"):
        if point_index >= len(artists):
            return None
        artist = artists[point_index]
        state = artist.get_visible()
        artist.set_visible(False)
        return lambda: artist.set_visible(state)
    if kind == "line" and len(artists) == 1:
        line = artists[0]
        x = np.asarray(line.get_xdata()).copy()
        y = np.asarray(line.get_ydata()).copy()
        if point_index >= len(x) or point_index >= len(y):
            return None
        changed_x, changed_y = x.copy(), y.copy()
        try:
            changed_y[point_index] = np.nan
        except (TypeError, ValueError):
            changed_y = changed_y.astype(float)
            changed_y[point_index] = np.nan
        line.set_data(changed_x, changed_y)
        return lambda: line.set_data(x, y)
    if kind == "scatter" and len(artists) == 1:
        collection = artists[0]
        offsets = np.ma.asarray(collection.get_offsets()).copy()
        if point_index >= len(offsets):
            return None
        changed = offsets.copy()
        changed[point_index] = [np.nan, np.nan]
        collection.set_offsets(changed)
        return lambda: collection.set_offsets(offsets)
    if kind == "heatmap" and len(artists) == 1:
        artist = artists[0]
        original = np.ma.asarray(artist.get_array()).copy()
        if point_index >= original.size:
            return None
        changed = np.ma.array(original, copy=True)
        mask = np.ma.getmaskarray(changed).copy().reshape(-1)
        mask[point_index] = True
        changed.mask = mask.reshape(changed.shape)
        artist.set_array(changed)
        annotations = _aligned_heatmap_annotations(entry, point_index)
        annotation_states = [(text, text.get_visible()) for text in annotations]
        for text, _ in annotation_states:
            text.set_visible(False)

        def restore_heatmap_point():
            for text, visible in annotation_states:
                text.set_visible(visible)
            artist.set_array(original)

        return restore_heatmap_point
    return None


def _color_property_mutation(entry, field, replacement):
    """Replace exactly one rendered color channel and return its restorer."""
    artists = [
        *(entry.get("artists") or []),
        *(entry.get("legend_artists") or []),
    ]
    kind = entry.get("kind")
    if field not in ("color", "colors", "edgecolor") or not artists:
        return None
    states = []
    try:
        for artist in artists:
            if field == "edgecolor":
                getter = getattr(artist, "get_edgecolor", None)
                setter = getattr(artist, "set_edgecolor", None)
            elif kind == "line":
                getter = getattr(artist, "get_color", None)
                setter = getattr(artist, "set_color", None)
            else:
                getter = getattr(artist, "get_facecolor", None)
                setter = getattr(artist, "set_facecolor", None)
            if not callable(getter) or not callable(setter):
                raise TypeError("target color property is not mutable")
            original = getter()
            try:
                original = original.copy()
            except AttributeError:
                pass
            states.append((setter, original))
            setter(replacement)
    except Exception:
        for setter, original in reversed(states):
            try:
                setter(original)
            except Exception:
                pass
        return None

    def restore_colors():
        for setter, original in reversed(states):
            setter(original)

    return restore_colors


def _text_property_mutation(entry, replacements):
    """Replace only target Text content while preserving its artist styling."""
    artists = entry.get("text_artists") or []
    if not artists:
        return None
    if isinstance(replacements, str):
        replacements = [replacements] * len(artists)
    if not isinstance(replacements, list) or len(replacements) != len(artists):
        return None
    states = []
    try:
        for artist, replacement in zip(artists, replacements):
            if not hasattr(artist, "get_text") or not hasattr(artist, "set_text"):
                raise TypeError("target text property is not mutable")
            states.append((artist, artist.get_text()))
            artist.set_text(replacement)
    except Exception:
        for artist, original in reversed(states):
            try:
                artist.set_text(original)
            except Exception:
                pass
        return None

    def restore_text():
        for artist, original in reversed(states):
            artist.set_text(original)

    return restore_text


def _legend_mapping_mutation(entry, labels, *, hide_handles):
    """Canonicalize only two legend mapping slots, leaving the frame intact."""
    items = entry.get("legend_items") or []
    if not isinstance(labels, list) or len(labels) != 2 or not items:
        return None
    selected = [
        (index, item) for index, item in enumerate(items)
        if item.get("label") in labels
    ]
    if len(selected) != 2 or {item.get("label") for _, item in selected} != set(labels):
        return None
    text_states = []
    handle_states = []
    try:
        for slot, item in selected:
            text = item.get("text")
            handle = item.get("handle")
            if text is None or handle is None:
                raise TypeError("target legend slot is incomplete")
            text_states.append((text, text.get_text()))
            text.set_text("ChartRepairBench Legend Slot %d" % slot)
            if hide_handles:
                handle_states.append((handle, handle.get_visible()))
                handle.set_visible(False)
    except Exception:
        for handle, visible in reversed(handle_states):
            try:
                handle.set_visible(visible)
            except Exception:
                pass
        for text, original in reversed(text_states):
            try:
                text.set_text(original)
            except Exception:
                pass
        return None

    def restore_legend_mapping():
        for handle, visible in reversed(handle_states):
            handle.set_visible(visible)
        for text, original in reversed(text_states):
            text.set_text(original)

    return restore_legend_mapping


def _target_contribution(fig, target, registry):
    """Encode the raster contribution of the exact semantic target."""
    if not isinstance(target, dict):
        return None
    canonical = json.dumps(target, sort_keys=True, separators=(",", ":"))
    target_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    mode = target.get("mode")
    restore = None
    states = []
    color_targets = []
    text_targets = []
    axis_counterfactual = None
    axis_gt_raw = None
    axis_flawed_raw = None
    try:
        visible = _canvas_rgb(fig)
        if mode == "point":
            root = target.get("series_root")
            entry = registry.get(root)
            if not isinstance(entry, dict):
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "target_series_not_rendered",
                }
            restore = _point_mutation(entry, target.get("point_index"))
            if restore is None:
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "target_point_isolation_unsupported",
                }
        elif mode == "color-properties":
            restorers = []
            seen = set()
            for path in target.get("paths") or []:
                if not isinstance(path, str) or "." not in path:
                    continue
                root, field = path.rsplit(".", 1)
                key = (root, field)
                entry = registry.get(root)
                if key in seen or not isinstance(entry, dict):
                    continue
                seen.add(key)
                callback = _color_property_mutation(
                    entry, field, (0.0, 0.0, 0.0, 0.0),
                )
                if callback is None:
                    for prior in reversed(restorers):
                        prior()
                    return {
                        "schema_version": "chartrepairbench.v2.target-contribution/1",
                        "supported": False,
                        "target_sha256": target_sha,
                        "reason": "target_color_property_isolation_unsupported",
                    }
                restorers.append(callback)
                color_targets.append((entry, field))
            if not restorers:
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "target_color_property_not_rendered",
                }

            def restore_colors():
                for callback in reversed(restorers):
                    callback()

            restore = restore_colors
        elif mode == "text-properties":
            restorers = []
            seen = set()
            for path in target.get("paths") or []:
                entry = registry.get(path)
                if path in seen or not isinstance(entry, dict):
                    continue
                seen.add(path)
                callback = _text_property_mutation(entry, "")
                if callback is None:
                    for prior in reversed(restorers):
                        prior()
                    return {
                        "schema_version": "chartrepairbench.v2.target-contribution/1",
                        "supported": False,
                        "target_sha256": target_sha,
                        "reason": "target_text_property_isolation_unsupported",
                    }
                restorers.append(callback)
                text_targets.append(entry)
            if not restorers:
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "target_text_property_not_rendered",
                }

            def restore_text():
                for callback in reversed(restorers):
                    callback()

            restore = restore_text
        elif mode == "legend-mapping":
            paths = target.get("paths") or []
            labels = target.get("labels")
            if len(paths) != 1 or not isinstance(paths[0], str):
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "target_legend_mapping_not_unique",
                }
            entry = registry.get(paths[0])
            restore = _legend_mapping_mutation(
                entry or {}, labels,
                hide_handles=target.get("mapping_mode") != "legend_order",
            )
            if restore is None:
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "target_legend_mapping_isolation_unsupported",
                }
        elif mode == "axis-clipping":
            axis_index = target.get("axis_index")
            axis_name = target.get("axis")
            roots = target.get("series_roots")
            clipped_points = target.get("clipped_points")
            paths = target.get("paths")
            if isinstance(axis_index, bool) or not isinstance(axis_index, int) or \
                    axis_index < 0 or axis_name not in ("x", "y") or \
                    not isinstance(roots, list) or not roots or \
                    len(set(roots)) != len(roots) or \
                    not isinstance(clipped_points, list) or not clipped_points or \
                    {
                        point.get("series_root")
                        for point in clipped_points if isinstance(point, dict)
                    } != set(roots) or \
                    paths != ["axes[%d].%slim" % (axis_index, axis_name)]:
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "axis_clipping_target_invalid",
                }
            axes = list(fig.get_axes())
            if axis_index >= len(axes):
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "axis_clipping_axis_not_rendered",
                }
            parsed_limits = []
            for name in ("gt_limits", "flawed_limits"):
                limits = target.get(name)
                if not isinstance(limits, list) or len(limits) != 2:
                    parsed_limits = []
                    break
                try:
                    pair = tuple(float(value) for value in limits)
                except (TypeError, ValueError):
                    parsed_limits = []
                    break
                if not all(math.isfinite(value) for value in pair) or pair[0] >= pair[1]:
                    parsed_limits = []
                    break
                parsed_limits.append(pair)
            if len(parsed_limits) != 2 or parsed_limits[0] == parsed_limits[1]:
                return {
                    "schema_version": "chartrepairbench.v2.target-contribution/1",
                    "supported": False,
                    "target_sha256": target_sha,
                    "reason": "axis_clipping_limits_invalid",
                }
            axis_object = axes[axis_index]
            getter = axis_object.get_xlim if axis_name == "x" else axis_object.get_ylim
            setter = axis_object.set_xlim if axis_name == "x" else axis_object.set_ylim
            original_limits = tuple(float(value) for value in getter())
            try:
                setter(*parsed_limits[0])
                visible = _canvas_rgb(fig)
            except Exception:
                setter(*original_limits)
                raise
            restorers = []
            seen_points = set()
            for point in clipped_points:
                root = point.get("series_root") if isinstance(point, dict) else None
                point_index = point.get("point_index") \
                    if isinstance(point, dict) else None
                key = (root, point_index)
                entry = registry.get(root)
                callback = None if key in seen_points or not isinstance(entry, dict) \
                    else _point_mutation(entry, point_index)
                if callback is None:
                    for prior in reversed(restorers):
                        prior()
                    setter(*original_limits)
                    return {
                        "schema_version": "chartrepairbench.v2.target-contribution/1",
                        "supported": False,
                        "target_sha256": target_sha,
                        "reason": "axis_clipped_mark_isolation_unsupported",
                    }
                seen_points.add(key)
                restorers.append(callback)

            def restore_axis_points():
                for callback in reversed(restorers):
                    callback()

            restore = restore_axis_points
            axis_counterfactual = (
                setter, original_limits, parsed_limits[1],
            )
        elif mode in ("artists", "chart-type"):
            artists = []
            for key in [*(target.get("series_roots") or []), *(target.get("paths") or [])]:
                entry = registry.get(key)
                if isinstance(entry, dict):
                    artists.extend(entry.get("artists") or [])
                    if mode == "chart-type":
                        artists.extend(entry.get("legend_artists") or [])
            artists = list({id(artist): artist for artist in artists}.values())
            states = [(artist, artist.get_visible()) for artist in artists]
            for artist, _ in states:
                artist.set_visible(False)

            def restore_artists():
                for artist, state in states:
                    artist.set_visible(state)

            restore = restore_artists
        else:
            return {
                "schema_version": "chartrepairbench.v2.target-contribution/1",
                "supported": False,
                "target_sha256": target_sha,
                "reason": "target_isolation_mode_unsupported",
            }

        hidden = _canvas_rgb(fig)
        contribution_hidden = hidden
        restore()
        restore = None
        neutralized_raw = None
        if mode == "axis-clipping":
            setter, original_limits, flawed_limits = axis_counterfactual
            axis_gt = hidden
            try:
                setter(*flawed_limits)
                axis_flawed = _canvas_rgb(fig)
            finally:
                setter(*original_limits)
            hidden = axis_gt
            neutralized_raw = axis_flawed.tobytes()
            axis_gt_raw = axis_gt.tobytes()
            axis_flawed_raw = neutralized_raw
        elif mode == "color-properties":
            neutral_restorers = []
            for entry, field in color_targets:
                callback = _color_property_mutation(
                    entry, field, (1.0, 0.0, 1.0, 1.0),
                )
                if callback is None:
                    raise TypeError("target color neutralization unsupported")
                neutral_restorers.append(callback)

            def restore_neutral_colors():
                for callback in reversed(neutral_restorers):
                    callback()

            restore = restore_neutral_colors
            neutralized_raw = _canvas_rgb(fig).tobytes()
            restore()
            restore = None
        elif mode == "text-properties":
            neutral_restorers = []
            for entry in text_targets:
                count = len(entry.get("text_artists") or [])
                replacements = [
                    "ChartRepairBench Target %d" % index
                    for index in range(count)
                ]
                callback = _text_property_mutation(entry, replacements)
                if callback is None:
                    raise TypeError("target text neutralization unsupported")
                neutral_restorers.append(callback)

            def restore_neutral_text():
                for callback in reversed(neutral_restorers):
                    callback()

            restore = restore_neutral_text
            neutralized_raw = _canvas_rgb(fig).tobytes()
            restore()
            restore = None
        elif mode == "legend-mapping":
            neutralized_raw = hidden.tobytes()
        contribution = visible.astype("int16") - contribution_hidden.astype("int16")
        raw = contribution.astype("<i2", copy=False).tobytes()
        hidden_raw = hidden.tobytes()
        return {
            "schema_version": "chartrepairbench.v2.target-contribution/1",
            "supported": True,
            "target_sha256": target_sha,
            "shape": list(contribution.shape),
            "encoding": "base64+zlib+int16le",
            "data": base64.b64encode(zlib.compress(raw, level=9)).decode("ascii"),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "rgb_encoding": "base64+zlib+uint8",
            "hidden_rgb_data": base64.b64encode(
                zlib.compress(hidden_raw, level=9)
            ).decode("ascii"),
            "hidden_rgb_sha256": hashlib.sha256(hidden_raw).hexdigest(),
            "neutralized_rgb_data": (
                base64.b64encode(
                    zlib.compress(neutralized_raw, level=9)
                ).decode("ascii")
                if neutralized_raw is not None else None
            ),
            "neutralized_rgb_sha256": (
                hashlib.sha256(neutralized_raw).hexdigest()
                if neutralized_raw is not None else None
            ),
            "axis_gt_rgb_sha256": (
                hashlib.sha256(axis_gt_raw).hexdigest()
                if axis_gt_raw is not None else None
            ),
            "axis_flawed_rgb_sha256": (
                hashlib.sha256(axis_flawed_raw).hexdigest()
                if axis_flawed_raw is not None else None
            ),
        }
    except Exception as exc:
        return {
            "schema_version": "chartrepairbench.v2.target-contribution/1",
            "supported": False,
            "target_sha256": target_sha,
            "reason": "%s: %s" % (type(exc).__name__, exc),
        }
    finally:
        if restore is not None:
            try:
                restore()
            except Exception:
                pass


# --------------------------------------------------------------------------- matplotlib

def _bars(ax):
    out = []
    for i, cont in enumerate(getattr(ax, "containers", [])):
        patches = list(getattr(cont, "patches", []) or [])
        if not patches or not all(
            _artist_visible(patch) and (
                _color_visible(patch.get_facecolor())
                or (patch.get_linewidth() > 0 and _color_visible(patch.get_edgecolor()))
            )
            for patch in patches
        ):
            continue
        vals, corners, vertical = [], [], True
        for p in patches:
            try:
                x, y = p.get_xy()
                w, h = p.get_width(), p.get_height()
            except Exception:
                continue
            corners.extend(((x, y), (x + w, y + h)))
            # matplotlib records orientation on the container for bar/barh
            if getattr(cont, "orientation", "vertical") == "horizontal":
                vertical = False
                vals.append([_num(y + h / 2.0), _num(w)])
            else:
                vals.append([_num(x + w / 2.0), _num(h)])
        if not vals:
            continue
        face = [_hex(p.get_facecolor()) for p in patches]
        item = {
            "kind": "bar" if vertical else "barh",
            "label": _label(cont, "bar%d" % i),
            "values": vals,
            "color": face[0] if face else None,
            "edgecolor": _hex(patches[0].get_edgecolor()),
            "axis_bounds": _axis_bounds(corners),
        }
        if len(set(face)) > 1:      # a per-bar palette is data, not decoration
            item["colors"] = face
        out.append(_with_capture_artists(item, patches))
    return out


def _axis_bounds(points):
    """Auxiliary geometry for axis containment, not another fidelity vector."""
    bounds = {}
    for index, letter in enumerate(("x", "y")):
        values = [value for point in points if len(point) > index
                  if (value := _num(point[index])) is not None]
        if values:
            bounds[letter] = [min(values), max(values)]
    return bounds


def _has_bar_container(ax):
    """True only for real bar charts: seaborn also gives box plots a container."""
    for cont in getattr(ax, "containers", []):
        ps = list(getattr(cont, "patches", []) or [])
        if ps and hasattr(ps[0], "get_xy"):   # Rectangle -> a bar
            return True
    return False


def _lines(ax):
    # seaborn and matplotlib draw error bars as short unlabelled Line2D objects
    # next to the bar containers; those are decoration, not series.
    has_bars = _has_bar_container(ax)
    out = []
    for i, ln in enumerate(getattr(ax, "lines", [])):
        if not _artist_visible(ln):
            continue
        try:
            xy = ln.get_xydata()
        except Exception:
            continue
        pts = [[_num(p[0]), _num(p[1])] for p in xy]
        pts = [p for p in pts if p[0] is not None or p[1] is not None]
        if not pts:
            continue
        if all(p[1] is None for p in pts):
            continue  # nothing actually plotted (empty confidence interval)
        unlabelled = _label(ln, None) is None
        if has_bars and unlabelled and len(pts) <= 3:
            continue
        marker = ln.get_marker()
        linestyle = ln.get_linestyle()
        line_visible = (
            linestyle not in (None, "None", "", " ")
            and ln.get_linewidth() > 0
            and _color_visible(ln.get_color())
        )
        marker_visible = (
            marker not in (None, "None", "", " ")
            and ln.get_markersize() > 0
            and (
                _color_visible(ln.get_markerfacecolor())
                or _color_visible(ln.get_markeredgecolor())
            )
        )
        if not line_visible and not marker_visible:
            continue
        out.append(_with_capture_artists({
            "kind": "line",
            "label": _label(ln, "line%d" % i),
            "values": pts,
            "color": _hex(ln.get_color()),
            "color_alpha": _rgba(ln.get_color())[3],
            "marker": (marker if marker not in (None, "None") else None),
            "linestyle": linestyle,
            "linewidth": _num(ln.get_linewidth()),
            "markersize": _num(ln.get_markersize()),
            "markerfacecolor": _hex(ln.get_markerfacecolor()),
            "markerfacecolor_alpha": _rgba(ln.get_markerfacecolor())[3],
            "markeredgecolor": _hex(ln.get_markeredgecolor()),
            "markeredgecolor_alpha": _rgba(ln.get_markeredgecolor())[3],
            "markeredgewidth": _num(ln.get_markeredgewidth()),
            "fillstyle": ln.get_fillstyle(),
            "drawstyle": ln.get_drawstyle(),
            "alpha": _num(ln.get_alpha()) if ln.get_alpha() is not None else 1.0,
            "zorder": _num(ln.get_zorder()),
        }, [ln]))
    return out


def _collections(ax):
    """scatter (PathCollection), stacked/filled areas (PolyCollection)."""
    out = []
    for i, col in enumerate(getattr(ax, "collections", [])):
        if not _artist_visible(col):
            continue
        name = type(col).__name__
        try:
            if name == "PathCollection":
                paths = col.get_paths()
                if not paths or not all(len(path.vertices) > 0 for path in paths):
                    continue
                offs = col.get_offsets()
                pts = [[_num(p[0]), _num(p[1])] for p in offs]
                if not pts:
                    continue
                sizes = [_num(v) for v in col.get_sizes()]
                if sizes and any(size is None or size <= 0 for size in sizes):
                    continue
                fc = [_hex(c) for c in col.get_facecolor()]
                visible_colors = [*col.get_facecolor(), *col.get_edgecolor()]
                if not visible_colors or not any(_color_visible(c) for c in visible_colors):
                    continue
                item = {
                    "kind": "scatter",
                    "label": _label(col, "scatter%d" % i),
                    "values": pts,
                    "color": fc[0] if fc else None,
                }
                if len(set(fc)) > 1:
                    item["colors"] = fc
                try:
                    if len(set(sizes)) > 1:   # size encodes a variable here
                        item["sizes"] = sizes
                except Exception:
                    pass
                out.append(_with_capture_artists(item, [col]))
            elif name in ("PolyCollection", "FillBetweenPolyCollection"):
                paths = col.get_paths()
                if not paths:
                    continue
                verts = paths[0].vertices
                # upper envelope of a filled band: first half of the ring
                half = max(1, len(verts) // 2)
                pts = [[_num(v[0]), _num(v[1])] for v in verts[:half]]
                fc = col.get_facecolor()
                if len(fc) and not any(_color_visible(color) for color in fc):
                    continue
                out.append(_with_capture_artists({
                    "kind": "area",
                    "label": _label(col, "area%d" % i),
                    "values": pts,
                    "color": _hex(fc[0]) if len(fc) else None,
                    "axis_bounds": _axis_bounds([vertex for path in paths for vertex in path.vertices]),
                }, [col]))
            elif name == "QuadMesh":
                arr = col.get_array()
                if arr is None:
                    continue
                flat = [_num(v) for v in list(getattr(arr, "compressed", arr.ravel)())]
                clim = [_num(v) for v in col.get_clim()]
                rendered = [_rgba(col.to_rgba(value)) for value in flat]
                if not rendered or not any(color and color[3] > 0 for color in rendered):
                    continue
                out.append(_with_capture_artists({
                    "kind": "heatmap",
                    "label": _label(col, "mesh%d" % i),
                    "values": [[j, v] for j, v in enumerate(flat)],
                    "color": None,
                    "clim": clim,
                    "rendered_colors": rendered,
                }, [col]))
        except Exception:
            continue
    return out


def _wedges(ax):
    try:
        from matplotlib.patches import Wedge
    except Exception:
        return []
    wedges = [p for p in getattr(ax, "patches", [])
              if isinstance(p, Wedge) and _artist_visible(p)]
    if not wedges:
        return []
    spans = []
    for w in wedges:
        try:
            spans.append(abs(w.theta2 - w.theta1))
        except Exception:
            spans.append(0.0)
    total = sum(spans) or 1.0
    vals, colors = [], []
    for i, (w, s) in enumerate(zip(wedges, spans)):
        # pie(labels=None) is common; fall back to the slice position
        vals.append([_label(w, None) or str(i), _num(s / total)])
        colors.append(_hex(w.get_facecolor()))
    item = {"kind": "pie", "label": "pie", "values": vals,
            "color": colors[0] if colors else None}
    if len(set(colors)) > 1:
        item["colors"] = colors
    return [_with_capture_artists(item, wedges)]


def _boxes(ax):
    """Box plots drawn with patch_artist / seaborn: the box is a PathPatch.

    Without this the whole chart reads as "no data", which would make every
    fidelity comparison vacuous for those instances.
    """
    if _has_bar_container(ax):
        return []                      # bar charts own their Rectangles
    try:
        from matplotlib.patches import PathPatch, Wedge
    except Exception:
        return []
    patches = [p for p in getattr(ax, "patches", [])
               if isinstance(p, PathPatch) and not isinstance(p, Wedge)
               and _artist_visible(p)]
    if not patches:
        return []
    vals, colors = [], []
    for i, p in enumerate(patches):
        try:
            ext = p.get_path().get_extents(p.get_patch_transform())
            x = _num((ext.x0 + ext.x1) / 2.0)
            vals.append([x, _num(ext.y0)])   # box bottom (lower quartile)
            vals.append([x, _num(ext.y1)])   # box top   (upper quartile)
            colors.append(_hex(p.get_facecolor()))
        except Exception:
            continue
    if not vals:
        return []
    item = {"kind": "box", "label": _label(patches[0], "box0"), "values": vals,
            "color": colors[0] if colors else None}
    if len(set(colors)) > 1:
        item["colors"] = colors
    return [_with_capture_artists(item, patches)]


def _images(ax):
    out = []
    for i, im in enumerate(getattr(ax, "images", [])):
        if not _artist_visible(im):
            continue
        try:
            arr = im.get_array()
            if arr is None:
                continue
            flat = [_num(v) for v in list(arr.ravel())]
            rendered = [_rgba(im.to_rgba(value)) for value in flat]
            if not rendered or not any(color and color[3] > 0 for color in rendered):
                continue
            out.append(_with_capture_artists({
                "kind": "heatmap",
                "label": _label(im, "image%d" % i),
                "values": [[j, v] for j, v in enumerate(flat)],
                "color": None,
                "clim": [_num(v) for v in im.get_clim()],
                "rendered_colors": rendered,
                "shape": list(getattr(arr, "shape", [])),
            }, [im]))
        except Exception:
            continue
    return out


def _annotations(ax):
    """Text annotations and arrow colours: summaries often describe them."""
    out = {}
    try:
        texts = [(t.get_text().strip(), _hex(t.get_color()))
                 for t in getattr(ax, "texts", [])
                 if (t.get_text() or "").strip() and _artist_visible(t)
                 and _color_visible(t.get_color())]
        if texts:
            out["annotations"] = [list(t) for t in texts]
    except Exception:
        pass
    try:
        # annotate(arrowprops=...) hangs its arrow off the Annotation itself,
        # not in ax.patches, so look there as well
        arrows = []
        for t in getattr(ax, "texts", []):
            ap = getattr(t, "arrow_patch", None)
            if ap is not None:
                arrows.append(_hex(ap.get_edgecolor()) or _hex(ap.get_facecolor()))
        from matplotlib.patches import FancyArrowPatch
        arrows += [_hex(p.get_edgecolor()) for p in getattr(ax, "patches", [])
                   if isinstance(p, FancyArrowPatch)]
        if arrows:
            out["arrow_colors"] = arrows
    except Exception:
        pass
    return out


def _axes_spec(ax, axis_index):
    if not _artist_visible(ax):
        return {"series": []}, {}
    series = _bars(ax) + _lines(ax) + _collections(ax) + _wedges(ax) + _boxes(ax) + _images(ax)
    registry = {}
    for index, item in enumerate(series):
        registry[_series_root(axis_index, series, index)] = {
            "artists": item.pop("_capture_artists", []),
            "kind": item.get("kind"),
            "label": item.get("label"),
            "annotation_artists": (
                list(getattr(ax, "texts", []))
                if item.get("kind") == "heatmap" else []
            ),
        }
    spec = {
        "series": series,
        "axis_sides": {"x": ax.xaxis.get_label_position(), "y": ax.yaxis.get_label_position()},
    }
    for name, artist in (
        ("title", ax.title),
        ("xlabel", ax.xaxis.label),
        ("ylabel", ax.yaxis.label),
    ):
        if name != "title" and (
            not ax.axison or not _artist_visible(getattr(ax, name[0] + "axis"))
        ):
            continue
        value = _text(artist)
        if value:
            spec[name] = value
            registry[f"axes[{axis_index}].{name}"] = {
                "artists": [artist],
                "text_artists": [artist],
            }
    spec.update(_annotations(ax))
    # summaries sometimes describe the colour of a label, not just its wording
    for name, artist in (("title_color", ax.title),
                         ("xlabel_color", ax.xaxis.label),
                         ("ylabel_color", ax.yaxis.label)):
        try:
            if name.removesuffix("_color") in spec:
                spec[name] = _hex(artist.get_color())
        except Exception:
            pass
    try:
        spec["xscale"], spec["yscale"] = ax.get_xscale(), ax.get_yscale()
    except Exception:
        pass
    # pie axes carry a meaningless unit square limit; skip those
    is_pie = any(s["kind"] == "pie" for s in series)
    if not is_pie:
        try:
            spec["xlim"] = [_num(v) for v in ax.get_xlim()]
            spec["ylim"] = [_num(v) for v in ax.get_ylim()]
        except Exception:
            pass
    try:
        for letter in ("x", "y"):
            axis = getattr(ax, letter + "axis")
            if not ax.axison or not _artist_visible(axis):
                continue
            spec[letter + "ticks"] = [_num(value) for value in axis.get_ticklocs()]
            spec[letter + "ticklabels"] = [t for t in (_text(t) for t in axis.get_ticklabels()) if t]
            spec[letter + "tick_policy"] = _tick_policy(axis)
    except Exception:
        pass
    try:
        leg = ax.get_legend()
        legend_items = _legend_items(leg)
        if legend_items:
            labels = [item["label"] for item in legend_items]
            spec["legend"] = {
                "labels": labels,
                "entries": _legend_entries(legend_items),
                "title": _text(leg.get_title()),
            }
            registry[f"axes[{axis_index}].legend.labels"] = {
                "artists": [leg],
                "text_artists": [item["text"] for item in legend_items],
                "legend_items": legend_items,
            }
            for item in legend_items:
                label, handle = item["label"], item["handle"]
                matches = [
                    entry for entry in registry.values()
                    if entry.get("label") == label
                ]
                if len(matches) == 1:
                    matches[0].setdefault("legend_artists", []).append(handle)
    except Exception:
        pass
    return spec, registry


def extract_matplotlib(fig, visibility_target=None):
    axes = []
    registry = {}
    for axis_index, ax in enumerate(fig.get_axes()):
        try:
            axis_spec, axis_registry = _axes_spec(ax, axis_index)
            axes.append(axis_spec)
            registry.update(axis_registry)
        except Exception as exc:  # never let one bad axes kill the capture
            axes.append({"error": "%s: %s" % (type(exc).__name__, exc), "series": []})
    spec = {"backend": "matplotlib", "axes": axes}
    try:
        spec["figure_size_inches"] = [_num(value) for value in fig.get_size_inches()]
        spec["figure_dpi"] = _num(fig.get_dpi())
    except Exception:
        pass
    try:
        if getattr(fig, "_suptitle", None) is not None:
            spec["suptitle"] = _text(fig._suptitle)
            registry["suptitle"] = {
                "artists": [fig._suptitle],
                "text_artists": [fig._suptitle],
            }
    except Exception:
        pass
    try:
        legends = [legend for legend in getattr(fig, "legends", [])
                   if _artist_visible(legend)]
        if legends:
            legend_items = _legend_items(legends[0])
            labels = [item["label"] for item in legend_items]
            spec["figlegend"] = labels
            spec["figlegend_entries"] = _legend_entries(legend_items)
            registry["figlegend.labels"] = {
                "artists": [legends[0]],
                "text_artists": [item["text"] for item in legend_items],
                "legend_items": legend_items,
            }
            for item in legend_items:
                label, handle = item["label"], item["handle"]
                matches = [
                    entry for entry in registry.values()
                    if entry.get("label") == label
                ]
                if len(matches) == 1:
                    matches[0].setdefault("legend_artists", []).append(handle)
    except Exception:
        pass
    if visibility_target is not None:
        spec["target_contribution"] = _target_contribution(
            fig, visibility_target, registry,
        )
    return spec


# --------------------------------------------------------------------------- plotly

_PLOTLY_KIND = {
    "bar": "bar", "scatter": "line", "scattergl": "line", "line": "line",
    "pie": "pie", "heatmap": "heatmap", "box": "box", "histogram": "bar",
    "scatterpolar": "line", "area": "area",
}


def _seq(v):
    """plotly hands back numpy arrays, whose truth value is ambiguous."""
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return []


def _plotly_color_visible(value):
    if not isinstance(value, str):
        return True
    normalized = value.strip().casefold()
    if normalized == "transparent":
        return False
    match = re.fullmatch(r"rgba\([^,]+,[^,]+,[^,]+,\s*([^)]+)\)", normalized)
    if match:
        try:
            return float(match.group(1)) > 0
        except ValueError:
            return False
    return True


def _positive_plotly_values(value, default):
    values = [value] if isinstance(value, (int, float)) else _seq(value)
    if not values:
        values = [default]
    numeric = [_num(item) for item in values]
    return bool(numeric) and all(item is not None and item > 0 for item in numeric)


def _plotly_series(trace, idx, layout=None):
    if trace.get("visible", True) in (False, "legendonly") or \
            _num(trace.get("opacity")) == 0:
        return None
    ttype = trace.get("type") or "scatter"
    kind = _PLOTLY_KIND.get(ttype, ttype)
    label = trace.get("name") or "%s%d" % (kind, idx)
    color = None
    marker = trace.get("marker") or {}
    if isinstance(marker.get("color"), str):
        color = marker["color"]
    elif isinstance((trace.get("line") or {}).get("color"), str):
        color = trace["line"]["color"]

    if ttype in {"scatter", "scattergl", "line"}:
        mode = trace.get("mode")
        if isinstance(mode, str):
            modes = set(mode.casefold().split("+"))
            line = trace.get("line") or {}
            line_visible = (
                "lines" in modes
                and (_num(line.get("width")) is None or _num(line.get("width")) > 0)
                and _plotly_color_visible(line.get("color"))
            )
            marker_visible = (
                "markers" in modes
                and _positive_plotly_values(marker.get("size"), 6)
                and _positive_plotly_values(marker.get("opacity"), 1)
                and _plotly_color_visible(marker.get("color"))
            )
            if not line_visible and not marker_visible:
                return None

    values = []
    if kind == "pie":
        labels = _seq(trace.get("labels"))
        vals = [v for v in (_num(v) for v in _seq(trace.get("values"))) if v is not None]
        total = sum(vals) or 1.0
        values = [[str(l), _num(v / total)] for l, v in zip(labels, vals)]
    elif kind == "heatmap":
        flat = []
        for row in _seq(trace.get("z")):
            flat.extend(_seq(row) if not isinstance(row, (int, float)) else [row])
        values = [[i, _num(v)] for i, v in enumerate(flat)]
    else:
        xs, ys = _seq(trace.get("x")), _seq(trace.get("y"))
        if not xs and ys:
            xs = list(range(len(ys)))
        for x, y in zip(xs, ys):
            nx = _num(x)
            values.append([nx if nx is not None else str(x), _num(y)])
    result = {"kind": kind, "label": label, "values": values, "color": color}
    if kind == "heatmap":
        numeric = [point[1] for point in values if point[1] is not None]
        coloraxis = (layout or {}).get(trace.get("coloraxis") or "") or {}
        zmin = _num(trace.get("zmin"))
        zmax = _num(trace.get("zmax"))
        if zmin is None:
            zmin = _num(coloraxis.get("cmin"))
        if zmax is None:
            zmax = _num(coloraxis.get("cmax"))
        if zmin is None and numeric:
            zmin = min(numeric)
        if zmax is None and numeric:
            zmax = max(numeric)
        result["clim"] = [zmin, zmax]
    return result


def _plotly_bar_extents(traces, layout):
    """Value-axis endpoints, including bases/stacking; no guessed bar widths."""
    out, stacks = {}, {}
    mode = layout.get("barmode") or "group"
    for index, trace in enumerate(traces):
        if trace.get("type") != "bar" or trace.get("visible", True) in (False, "legendonly") \
                or _num(trace.get("opacity")) == 0:
            continue
        horizontal = trace.get("orientation") == "h"
        value_axis, category_axis = ("x", "y") if horizontal else ("y", "x")
        unavailable = [category_axis]
        values = _seq(trace.get(value_axis))
        categories = _seq(trace.get(category_axis)) or list(range(len(values)))
        numeric = [_num(value) for value in values]
        if (layout.get("barnorm") or any(value is None for value in numeric)
                or len(categories) != len(values)
                or trace.get("xaxis", "x") != "x" or trace.get("yaxis", "y") != "y"
                or (mode == "stack" and any(value < 0 for value in numeric))):
            out[index] = ({}, ["x", "y"])
            continue
        raw_base = trace.get("base")
        bases = _seq(raw_base) if raw_base is not None and _num(raw_base) is None else [raw_base] * len(values)
        if len(bases) != len(values):
            out[index] = ({}, ["x", "y"])
            continue
        endpoints = []
        for category, value, base in zip(categories, numeric, bases):
            if raw_base is None and mode in {"stack", "relative"}:
                sign = -1 if mode == "relative" and value < 0 else 1
                key = (value_axis, str(trace.get("offsetgroup", "")), str(category), sign)
                start = stacks.get(key, 0.0)
                stacks[key] = start + value
            else:
                start = 0.0 if base is None else _num(base)
            if start is None:
                unavailable = ["x", "y"]
                break
            endpoints.extend([start, start + value])
        bounds = {value_axis: [min(endpoints), max(endpoints)]} if endpoints else {}
        out[index] = (bounds, unavailable)
    return out


def extract_plotly(fig):
    d = fig.to_plotly_json() if hasattr(fig, "to_plotly_json") else dict(fig)
    layout = d.get("layout") or {}

    def _t(node):
        if isinstance(node, dict):
            if not _plotly_color_visible((node.get("font") or {}).get("color")):
                return None
            v = node.get("text")
            return v.strip() if isinstance(v, str) and v.strip() else None
        return node.strip() if isinstance(node, str) and node.strip() else None

    ax = {
        "title": _t(layout.get("title")),
        "xlabel": (_t((layout.get("xaxis") or {}).get("title"))
                   if (layout.get("xaxis") or {}).get("visible") is not False else None),
        "ylabel": (_t((layout.get("yaxis") or {}).get("title"))
                   if (layout.get("yaxis") or {}).get("visible") is not False else None),
        "series": [series for i, trace in enumerate(d.get("data") or [])
                   if (series := _plotly_series(trace, i, layout)) is not None],
    }
    bar_extents = _plotly_bar_extents(d.get("data") or [], layout)
    visible_index = 0
    for index, trace in enumerate(d.get("data") or []):
        if _plotly_series(trace, index, layout) is None:
            continue
        if index in bar_extents:
            bounds, unavailable = bar_extents[index]
            ax["series"][visible_index].update(axis_bounds=bounds, axis_bounds_unavailable=unavailable)
        visible_index += 1
    for src, dst in (("xaxis", "xlim"), ("yaxis", "ylim")):
        rng = (layout.get(src) or {}).get("range")
        if rng:
            ax[dst] = [_num(v) for v in rng]
    for src, dst in (("xaxis", "xscale"), ("yaxis", "yscale")):
        typ = (layout.get(src) or {}).get("type")
        if typ:
            ax[dst] = "log" if typ == "log" else "linear"
    for src, dst in (("xaxis", "xticklabels"), ("yaxis", "yticklabels")):
        axis = layout.get(src) or {}
        tt = axis.get("ticktext")
        if (tt and axis.get("visible") is not False
                and axis.get("showticklabels") is not False
                and _plotly_color_visible((axis.get("tickfont") or {}).get("color"))):
            ax[dst] = [str(v) for v in tt]
    for src, dst in (("xaxis", "xticks"), ("yaxis", "yticks")):
        values = (layout.get(src) or {}).get("tickvals")
        if values:
            ax[dst] = [_num(value) for value in values]
    labels = [s["label"] for i, trace in enumerate(d.get("data") or [])
              if trace.get("showlegend") is not False
              and (s := _plotly_series(trace, i, layout)) is not None and s.get("label")]
    legend_requested = (layout.get("showlegend") is True
                        or any(trace.get("showlegend") is True for trace in d.get("data") or []))
    if (layout.get("showlegend") is not False and (len(labels) > 1 or (labels and legend_requested))
            and _plotly_color_visible(((layout.get("legend") or {}).get("font") or {}).get("color"))):
        ax["legend"] = {"labels": labels, "title": _t((layout.get("legend") or {}).get("title"))}
    spec = {"backend": "plotly", "axes": [ax]}
    width, height = _num(layout.get("width")), _num(layout.get("height"))
    if width is not None or height is not None:
        spec["canvas_size_px"] = [width, height]
    return spec


def dump(spec, path="figure.json"):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False)
