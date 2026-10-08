"""Zone definitions, loading, rendering, and geometric spatial containment logic."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Distinct colors in BGR for rendering multiple zones
ZONE_COLORS = [
    (50, 205, 50),    # Lime green
    (255, 140, 0),    # Deep sky blue / orange
    (238, 130, 238),  # Violet
    (0, 215, 255),    # Gold / yellow
    (255, 105, 180),  # Hot pink
    (0, 165, 255),    # Bright orange
    (255, 255, 0),    # Cyan
]
RESTRICTED_COLOR = (40, 40, 230)  # Crimson red for restricted zones


def footpoint_of_box(bbox: tuple[float, float, float, float] | Sequence[float]) -> tuple[float, float]:
    """Compute footpoint of a bounding box as bottom-center: ((x1 + x2) / 2, y2)."""
    if len(bbox) < 4:
        raise ValueError(f"bbox must contain at least 4 coordinates (x1, y1, x2, y2), got {bbox}")
    x1, y1, x2, y2 = bbox[:4]
    return ((float(x1) + float(x2)) / 2.0, float(y2))


# Alias for backward compatibility / convenience
box_footpoint = footpoint_of_box


@dataclass(frozen=True)
class Zone:
    """A spatial polygon zone within a camera's field of view."""

    name: str
    polygon: tuple[tuple[float, float], ...]
    zone_type: str = "normal"

    def contains_point(self, point: tuple[float, float]) -> bool:
        """Check if (x, y) point is inside or on the boundary of the polygon."""
        if not self.polygon or len(self.polygon) < 3:
            return False
        poly_arr = np.array(self.polygon, dtype=np.float32)
        dist = cv2.pointPolygonTest(poly_arr, (float(point[0]), float(point[1])), measureDist=False)
        return dist >= 0

    def bbox_bottom_center(
        self,
        bbox: tuple[float, float, float, float] | Sequence[float],
    ) -> tuple[float, float]:
        """Compute the bottom-center anchor of a bounding box (ground contact point)."""
        return footpoint_of_box(bbox)

    def contains_bbox(
        self,
        bbox: tuple[float, float, float, float] | Sequence[float],
        anchor: str = "bottom_center",
    ) -> bool:
        """Check if a bounding box intersects/is contained in the zone based on anchor point."""
        if anchor == "bottom_center":
            pt = self.bbox_bottom_center(bbox)
        elif anchor == "center":
            x1, y1, x2, y2 = bbox[:4]
            pt = ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
        else:
            raise ValueError(f"Unsupported anchor type: {anchor}")

        return self.contains_point(pt)

    def scale(self, scale_x: float, scale_y: float) -> Zone:
        """Return a new Zone with coordinates scaled by scale_x and scale_y."""
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError(f"Scale factors must be positive, got ({scale_x}, {scale_y})")
        scaled_polygon = tuple((float(x * scale_x), float(y * scale_y)) for x, y in self.polygon)
        return Zone(name=self.name, polygon=scaled_polygon, zone_type=self.zone_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize zone to dictionary format."""
        return {
            "name": self.name,
            "type": self.zone_type,
            "polygon": [[round(x, 2), round(y, 2)] for x, y in self.polygon],
        }


@dataclass(frozen=True)
class CameraZoneConfig:
    """Camera zone configuration containing camera ID, reference resolution, and zones."""

    camera_id: str
    frame_width: int
    frame_height: int
    zones: list[Zone]


def zone_of_point(zones: Sequence[Zone], x: float, y: float) -> list[str]:
    """Return list of names of all zones containing point (x, y) using cv2.pointPolygonTest.

    A point can be in several zones if polygons overlap.
    Points on boundaries/vertices are considered inside (dist >= 0).
    """
    pt = (float(x), float(y))
    return [z.name for z in zones if z.contains_point(pt)]


def zone_of_bbox(
    zones: Sequence[Zone],
    bbox: tuple[float, float, float, float] | Sequence[float],
) -> list[str]:
    """Return list of zone names containing the footpoint ((x1+x2)/2, y2) of the bounding box."""
    fx, fy = footpoint_of_box(bbox)
    return zone_of_point(zones, fx, fy)


def scale_zones(
    zones: Sequence[Zone],
    orig_width: int,
    orig_height: int,
    target_width: int,
    target_height: int,
) -> list[Zone]:
    """Scale all zones from an original resolution to a target resolution."""
    if orig_width <= 0 or orig_height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Resolutions must be positive: orig=({orig_width}, {orig_height}), target=({target_width}, {target_height})"
        )
    if orig_width == target_width and orig_height == target_height:
        return list(zones)

    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    return [z.scale(scale_x, scale_y) for z in zones]


def parse_zones_from_config(zone_dicts: Sequence[dict[str, Any]]) -> list[Zone]:
    """Parse list of raw dictionary configurations into Zone objects."""
    zones = []
    for item in zone_dicts:
        polygon = tuple((float(p[0]), float(p[1])) for p in item["polygon"])
        zones.append(
            Zone(
                name=item["name"],
                polygon=polygon,
                zone_type=item.get("type", "normal"),
            )
        )
    return zones


def load_zone_config(
    source: str | Path | dict[str, Any],
    target_width: int | None = None,
    target_height: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> CameraZoneConfig:
    """Load camera zone configuration from a JSON/YAML file or dictionary.

    If target resolution is supplied (either via target_size=(w, h) or
    target_width and target_height), polygon coordinates are automatically scaled
    if the reference frame dimensions differ from the target dimensions.
    """
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Zone config file not found: {source_path}")
        with open(source_path, "r", encoding="utf-8") as f:
            if source_path.suffix.lower() in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError(f"Expected path or dict, got {type(source)}")

    camera_id = data.get("camera_id", "unknown")
    frame_width = int(data.get("frame_width", 0))
    frame_height = int(data.get("frame_height", 0))
    raw_zones = data.get("zones", [])

    zones = parse_zones_from_config(raw_zones)

    if target_size is not None:
        target_width, target_height = target_size

    if (
        target_width is not None
        and target_height is not None
        and frame_width > 0
        and frame_height > 0
        and (frame_width != target_width or frame_height != target_height)
    ):
        logger.info(
            "Scaling zones for camera '%s' from %dx%d to %dx%d",
            camera_id,
            frame_width,
            frame_height,
            target_width,
            target_height,
        )
        zones = scale_zones(zones, frame_width, frame_height, target_width, target_height)
        frame_width = target_width
        frame_height = target_height

    return CameraZoneConfig(
        camera_id=camera_id,
        frame_width=frame_width,
        frame_height=frame_height,
        zones=zones,
    )


def load_zones(
    source: str | Path | dict[str, Any],
    target_width: int | None = None,
    target_height: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> list[Zone]:
    """Load list of Zone objects from JSON/YAML or dict, with optional resolution scaling."""
    config = load_zone_config(
        source,
        target_width=target_width,
        target_height=target_height,
        target_size=target_size,
    )
    return config.zones


def save_zones(
    camera_id: str,
    frame_width: int,
    frame_height: int,
    zones: Sequence[Zone | dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Save zone definitions to configs/zones_<camera_id>.json."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    formatted_zones = []
    for z in zones:
        if isinstance(z, Zone):
            formatted_zones.append(z.to_dict())
        elif isinstance(z, dict):
            formatted_zones.append({
                "name": z["name"],
                "type": z.get("type", "normal"),
                "polygon": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in z["polygon"]],
            })
        else:
            raise TypeError(f"Expected Zone or dict, got {type(z)}")

    payload = {
        "camera_id": camera_id,
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "zones": formatted_zones,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved %d zones to %s", len(formatted_zones), out_file)
    return out_file


def draw_zones_on_frame(
    frame: np.ndarray,
    zones: Sequence[Zone],
    alpha: float = 0.25,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw zones with semi-transparent fill, colored boundary, and label badges."""
    annotated = frame.copy()
    overlay = frame.copy()

    for idx, zone in enumerate(zones):
        if not zone.polygon or len(zone.polygon) < 3:
            continue
        pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))

        # Color selection: restricted zones get crimson red, normal get rotating palette
        if zone.zone_type.lower() == "restricted":
            color = RESTRICTED_COLOR
        else:
            color = ZONE_COLORS[idx % len(ZONE_COLORS)]

        cv2.fillPoly(overlay, [pts], color=color)

    # Blend filled polygons
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

    # Draw solid boundary lines and text badges on top
    for idx, zone in enumerate(zones):
        if not zone.polygon or len(zone.polygon) < 3:
            continue
        pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        color = RESTRICTED_COLOR if zone.zone_type.lower() == "restricted" else ZONE_COLORS[idx % len(ZONE_COLORS)]

        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # Label position: top-left bounding box of the polygon
        xs = [p[0] for p in zone.polygon]
        ys = [p[1] for p in zone.polygon]
        label_x = int(round(min(xs)))
        label_y = int(round(min(ys))) - 6
        if label_y < 20:
            label_y = int(round(min(ys))) + 20

        label_text = f"{zone.name} ({zone.zone_type})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Background badge for text
        cv2.rectangle(
            annotated,
            (label_x, label_y - txt_h - 4),
            (label_x + txt_w + 6, label_y + baseline),
            (20, 20, 20),
            -1,
        )
        # Text in zone color
        cv2.putText(
            annotated,
            label_text,
            (label_x + 3, label_y - 2),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return annotated
