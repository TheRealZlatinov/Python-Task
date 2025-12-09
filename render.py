import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import sys
import time
from typing import List, Dict, Any, Tuple

# ---------------------------------------
# Utility: 4x4 Transform Builders (with type hints)
# ---------------------------------------
def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """Returns a 4x4 translation matrix."""
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T

def rotation_z_matrix(deg: float) -> np.ndarray:
    """Returns a 4x4 rotation matrix around the Z-axis (in degrees)."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.eye(4)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R

# ---------------------------------------
# OOP Abstraction: The Part Class
# ---------------------------------------
class Part:
    """
    Represents a single geometric part (a cuboid) in the 3D scene.
    Encapsulates dimension, transformation, and vertex calculation.
    """
    def __init__(self, index: int, part_data: Dict[str, Any], T_extra: np.ndarray):
        self.name = part_data.get("name", f"Part{index}")
        self.width = part_data.get("width", 1.0)
        self.depth = part_data.get("depth", 1.0)
        self.height = part_data.get("height", 1.0)
        self.T_extra = T_extra
        
        # Robustly load the ecsBox (Essential Coordinate System) matrix
        try:
            # The matrix is expected to be flattened, 1D array in Fortran (column-major) order.
            self.ecs = np.array(part_data["ecsBox"]).reshape((4, 4), order="F")
        except Exception as e:
            # Raise a custom error to be caught in the main loop
            raise ValueError(f"'{self.name}' invalid ecsBox format or missing key. Error: {e}")
            
        self.transformed_vertices = self._calculate_vertices()
        
    def _calculate_vertices(self) -> np.ndarray:
        """
        Calculates the 8 world-space vertices of the cuboid after applying
        the part's intrinsic transformation (ECS) and global extra transformation.
        """
        hw, hd, hh = self.width / 2, self.depth / 2, self.height / 2

        # Local cube vertices (8 points, 4 dimensions [x, y, z, 1])
        local_vertices = np.array([
            [-hw, -hd, -hh, 1], [ hw, -hd, -hh, 1], [ hw,  hd, -hh, 1], [-hw,  hd, -hh, 1],
            [-hw, -hd,  hh, 1], [ hw, -hd,  hh, 1], [ hw,  hd,  hh, 1], [-hw,  hd,  hh, 1],
        ], dtype=float)

        # The final transformation matrix (ECS applied first, then global extra transform)
        T_final = self.ecs @ self.T_extra
        
        # Apply transform: (4x4 @ 8x4.T) -> 4x8 -> take first 3 rows -> 3x8.T -> 8x3 array of (x, y, z)
        # This is a highly efficient numpy operation.
        return (T_final @ local_vertices.T)[:3].T
    
    def get_faces(self) -> List[np.ndarray]:
        """Returns a list of the 6 faces, where each face is a list of 4 vertices."""
        v = self.transformed_vertices
        # Define the 6 faces using the 8 vertices (indices 0-7)
        faces = [
            [v[0], v[1], v[2], v[3]],  # Bottom
            [v[4], v[5], v[6], v[7]],  # Top
            [v[0], v[1], v[5], v[4]],  # Side 1
            [v[1], v[2], v[6], v[5]],  # Side 2
            [v[2], v[3], v[7], v[6]],  # Side 3
            [v[3], v[0], v[4], v[7]],  # Side 4
        ]
        return faces

# ---------------------------------------
# Rendering Function (Refactored)
# ---------------------------------------
def render_scaffold(args: argparse.Namespace) -> None:
    """Main rendering function, now using the Part class."""
    start_time = time.time() if args.performance else None

    print(f"Loading data from {args.file}...")
    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"FATAL: Cannot load JSON: {e}")
        sys.exit(1)

    # Improved validation for the required structure
    if not isinstance(data.get("parts"), list):
        print("FATAL: JSON must contain a top-level array named 'parts'")
        sys.exit(1)

    # --- Setup Plot ---
    fig = plt.figure(figsize=(12, 12))
    projection_type = 'ortho' if args.view_ortho else 'persp'
    ax = fig.add_subplot(111, projection='3d', proj_type=projection_type)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Render: {args.file}')

    # --- Global Transforms ---
    T_extra = np.eye(4)
    if args.rz is not None:
        T_extra = rotation_z_matrix(args.rz) @ T_extra
    
    # NOTE: The order of applying translations/rotations matters for T_extra.
    # Here, translation is applied after rotation relative to the object's local frame.
    if args.tx or args.ty or args.tz:
        T_extra = translation_matrix(args.tx, args.ty, args.tz) @ T_extra

    # --- Part Processing ---
    parts_rendered = 0
    # Use a list of numpy arrays for efficient aggregation later
    all_coords: List[np.ndarray] = [] 

    for i, part_data in enumerate(data["parts"]):
        try:
            # Instantiating the Part class handles validation and vertex calculation
            part = Part(i, part_data, T_extra)
        except ValueError as e:
            print(f"Warning: {e} → skipped")
            continue

        # Filter (-b)
        if args.scaffolding_only and "ScaffoldingBox" not in part.name:
            continue
        
        # Collect vertices efficiently using numpy array append
        all_coords.append(part.transformed_vertices)

        # Determine rendering styles based on name/flags
        is_scaffold = "ScaffoldingBox" in part.name
        
        if args.highlight and args.highlight in part.name:
            facecolor, edgecolor, lw, alpha = "red", "red", 2.0, 0.45
        else:
            facecolor = "skyblue" if is_scaffold else "darkred"
            edgecolor = "black"
            lw = 0.5
            alpha = 0.25

        # Add the faces (Poly3DCollection expects a list of face vertex lists)
        ax.add_collection3d(Poly3DCollection(
            part.get_faces(),
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=lw,
            alpha=alpha
        ))

        parts_rendered += 1

    # --- Bounding Box and Display ---
    if not all_coords:
        print("No parts rendered.")
        return

    # Efficiently stack all vertex arrays into a single coordinate array
    ac = np.vstack(all_coords) 
    
    # Auto bounds based on all rendered coordinates
    max_range = max(np.ptp(ac[:,0]), np.ptp(ac[:,1]), np.ptp(ac[:,2])) / 2
    mid = ac.mean(axis=0)

    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    ax.set_box_aspect([1,1,1])

    # Camera direction overrides
    if args.view_x: ax.view_init(elev=0, azim=0)
    if args.view_y: ax.view_init(elev=0, azim=90)
    if args.view_z: ax.view_init(elev=90, azim=0)

    print(f"Rendered parts: {parts_rendered}")

    if args.performance:
        print(f"Time: {time.time() - start_time:.6f} sec")

    plt.show()

    # Write output file (-o)
    if args.output:
        with open(args.output, "w") as f:
            # Note: Dumping the original input data 'data', not the processed Part objects
            json.dump(data, f, indent=2) 
        print(f"Output saved to {args.output}")


# ---------------------------------------
# COMMAND LINE PARSER
# ---------------------------------------
def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    p = argparse.ArgumentParser(description="Render scaffolding JSON")

    p.add_argument("-f", "--file", required=True, help="Input JSON file")
    p.add_argument("-o", "--output", help="Output JSON file")

    # highlight flag (-H)
    p.add_argument("-H", "--highlight", help="Highlight part by name (substring match)")

    p.add_argument("-b", "--scaffolding-only", action="store_true",
                   help="Render only ScaffoldingBox items")

    p.add_argument("-tx", type=float, default=0.0, help="Translate X offset")
    p.add_argument("-ty", type=float, default=0.0, help="Translate Y offset")
    p.add_argument("-tz", type=float, default=0.0, help="Translate Z offset")
    p.add_argument("-rz", type=float, help="Rotate by Z axis (degrees)")

    p.add_argument("-vo", "--view-ortho", action="store_true", help="Orthographic view (no perspective distortion)")
    p.add_argument("-vx", "--view-x", action="store_true", help="View along X-axis")
    p.add_argument("-vy", "--view-y", action="store_true", help="View along Y-axis")
    p.add_argument("-vz", "--view-z", action="store_true", help="View along Z-axis")

    p.add_argument("-p", "--performance", action="store_true",
                   help="Measure performance time")

    return p.parse_args()


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    args = parse_args()
    render_scaffold(args)