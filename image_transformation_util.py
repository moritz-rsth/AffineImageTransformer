import torch
from typing import Tuple, List
import cv2
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
import json


class LinearBoundedSector:
    """
    First we need to define a class modeling a Sector.
    A sector represents one angualar piece of the image,
    e.g. the abdominal wall in our surgery example.
    """
    
    def __init__(
            self, 
            center: Tuple[int, int], 
            edge_point1: Tuple[int, int], 
            edge_point2: Tuple[int, int], 
            bound_width : int, 
            bound_height : int):
        """
        Inititalizes the Linear Sector.
        A linear sector is represented by the follwoing attributes:
            - center: Tuple[int, int] marks the center of the outgoing rays
            - edge_point1: Tuple[int, int] marks the first edge point (clockwise)
            - edge_point2: Tuple[int, int] marks the second edge point (clockwise)
        """
        self.center = center
        self.edge_point1 = edge_point1
        self.edge_point2 = edge_point2
        self.bound_width = bound_width
        self.bound_height = bound_height
        
    @staticmethod
    def get_sectors_from_raw_data(
            raw_points, 
            bound_width: int, 
            bound_height: int):
        """
        Specifically gives you the three sector from the raw data points.
            - raw_points: List of 4 torch tensors containing relative coordinates (normalized 0-1)
            points = [
                torch.tensor([x_center, y_center]),  # Center (relative coordinates, will be converted to absolute)
                torch.tensor([x_adhesion, y_adhesion]),  # Direction of adhesion (relative coordinates)
                torch.tensor([x_left, y_left]),  # Direction of left mounting (relative coordinates)
                torch.tensor([x_right, y_right]),  # Direction of right mounting (relative coordinates)
            ]
            All coordinates are converted from relative (0-1 range) to absolute pixel coordinates
            by multiplying by bound_width and bound_height.
        returns sector_abdominal_wall, sector_attached_tissue, sector_detached_tissue
        """
        if(len(raw_points) != 4):
            raise ValueError(f"Invalid raw_points format: expected 4 points, got {len(raw_points)}")

        # Convert relative coordinates to absolute pixel coordinates
        # Center coordinates (relative [0-1] -> absolute pixels)
        center_rel = raw_points[0].float().tolist()
        center = [int(center_rel[0] * bound_width), int(center_rel[1] * bound_height)]
        
        # Direction coordinates (relative [0-1] -> absolute pixels)
        left_mounting_direction_rel = raw_points[1].float().tolist()  # ABDOMINAL WALL true
        left_mounting_direction = [int(left_mounting_direction_rel[0] * bound_width), 
                                int(left_mounting_direction_rel[1] * bound_height)]

        right_mounting_direction_rel = raw_points[2].float().tolist() # ABDOMINAL WALL false  
        right_mounting_direction = [int(right_mounting_direction_rel[0] * bound_width), 
                                int(right_mounting_direction_rel[1] * bound_height)]

        adhesion_direction_rel = raw_points[3].float().tolist()  # FAT
        adhesion_direction = [int(adhesion_direction_rel[0] * bound_width), 
                            int(adhesion_direction_rel[1] * bound_height)]
        
        adhesion = LinearBoundedSector._edge_intersect(
            center=center,
            edge_direction=adhesion_direction,
            bound_height=bound_height,
            bound_width=bound_width,
        )

        left_mounting = LinearBoundedSector._edge_intersect(
            center=center,
            edge_direction=left_mounting_direction,
            bound_height=bound_height,
            bound_width=bound_width,
        )

        right_mounting = LinearBoundedSector._edge_intersect(
            center=center,
            edge_direction=right_mounting_direction,
            bound_height=bound_height,
            bound_width=bound_width,
        )

        sector_abdominal_wall = LinearBoundedSector(center=center, edge_point1=left_mounting, edge_point2=right_mounting, bound_height=bound_height, bound_width=bound_width)
        sector_attached_tissue = LinearBoundedSector(center=center, edge_point1=adhesion, edge_point2=left_mounting, bound_height=bound_height, bound_width=bound_width)
        sector_detached_tissue = LinearBoundedSector(center=center, edge_point1=right_mounting, edge_point2=adhesion, bound_height=bound_height, bound_width=bound_width)

        return sector_abdominal_wall, sector_attached_tissue, sector_detached_tissue
    
    @staticmethod
    def _edge_intersect( 
            center: Tuple[int, int], 
            edge_direction: Tuple[float, float], 
            bound_width: int, 
            bound_height: int):
        """
        Compute intersection of a ray with the image border.
        Parameters
        - center: Tuple (x, y) for the center point.
        - edge_direction: Tuple (x, y) for the edge point.
        - width/height: Image dimensions in pixels.
        Returns
        - (x, y) integer pixel coordinate of the nearest intersection along the ray.
        """
        
        x_d, y_d = center
        x_i, y_i = edge_direction

        dx = x_i - x_d
        dy = y_i - y_d

        candidates = []

        # left edge
        if dx != 0:
            k = (0 - x_d) / dx
            y = y_d + k * dy
            if 0 <= y <= bound_height:
                candidates.append((k, 0, y))

        # right edge x=width
        if dx != 0:
            k = (bound_width - x_d) / dx
            y = y_d + k * dy
            if 0 <= y <= bound_height:
                candidates.append((k, bound_width, y))

        # upper edge y=0
        if dy != 0:
            k = (0 - y_d) / dy
            x = x_d + k * dx
            if 0 <= x <= bound_width:
                candidates.append((k, x, 0))

        # lower edge y=height
        if dy != 0:
            k = (bound_height - y_d) / dy
            x = x_d + k * dx
            if 0 <= x <= bound_width:
                candidates.append((k, x, bound_height))

        # smallest possible k
        candidates = [(k, x, y) for k, x, y in candidates if k >= 0]
        if not candidates:
            raise RuntimeError(f"Could not find intersection of ray from center {center} in direction {edge_direction} with image bounds ({bound_width}x{bound_height})")

        k, x_e, y_e = min(candidates, key=lambda t: t[0])
        return int(x_e), int(y_e)

    def add_angle_to_sector(self, angle: int, start: bool):
        """
        Broadens/narrows the sector via the specified angle.
        
        Args:
            angle: Angle in degrees to add
                - Positive = clockwise rotation
                - Negative = counterclockwise rotation
            start: If True, adjust edge_point1 (first edge clockwise from center)
                If False, adjust edge_point2 (second edge clockwise from center)
        """
        # Select the point to update
        point_to_be_updated = self.edge_point1 if start else self.edge_point2
        
        # Calculate current direction vector from center to edge point
        direction_x = point_to_be_updated[0] - self.center[0]
        direction_y = point_to_be_updated[1] - self.center[1]
        
        # Calculate current angle in radians
        current_angle = math.atan2(direction_y, direction_x)
        
        # Convert degrees to radians
        angle_radians = math.radians(angle)
        new_angle = current_angle + angle_radians
        
        # Calculate new direction vector (unit vector)
        direction_unit_x = math.cos(new_angle)
        direction_unit_y = math.sin(new_angle)
        
        # Create a point far enough from center in the new direction
        # Use a distance larger than the image diagonal to ensure intersection
        max_distance = math.sqrt(self.bound_width**2 + self.bound_height**2) * 2
        far_point = (
            int(self.center[0] + direction_unit_x * max_distance),
            int(self.center[1] + direction_unit_y * max_distance)
        )
        
        # Find intersection with image bounds using the far point
        updated_point = self._edge_intersect(
            self.center, 
            far_point, 
            self.bound_width, 
            self.bound_height
        )
        
        # Update the appropriate edge point
        if start:
            self.edge_point1 = updated_point
        else:
            self.edge_point2 = updated_point

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [0, 2π) range.
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle in [0, 2π)
        """
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle
    
    def _get_angle_from_center(self, point: Tuple[int, int]) -> float:
        """
        Get normalized angle from center to point.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            Normalized angle in [0, 2π)
        """
        angle = math.atan2(point[1] - self.center[1], point[0] - self.center[0])
        return self._normalize_angle(angle)
    
    def _get_point_at_angle(self, angle: float) -> Tuple[int, int]:
        """
        Get point on image boundary at given angle from center.
        
        Args:
            angle: Angle in radians (normalized)
            
        Returns:
            (x, y) coordinates of intersection point with image boundary
        """
        # Create a point far enough from center in the direction of the angle
        max_distance = math.sqrt(self.bound_width**2 + self.bound_height**2) * 2
        direction_x = math.cos(angle)
        direction_y = math.sin(angle)
        far_point = (
            int(self.center[0] + direction_x * max_distance),
            int(self.center[1] + direction_y * max_distance)
        )
        return self._edge_intersect(self.center, far_point, self.bound_width, self.bound_height)

    def get_radial_triangles(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns always three radial triangles.
        
        Simplified approach:
        1. Collect all relevant points along the sector boundary
        2. Sort them by angle (clockwise from edge_point1)
        3. Create triangles: [center, point_i, point_i+1]
        
        Returns:
            Three triangles, each as a list of three (x, y) coordinates.
            All triangles have center as first point.
        """
        # Get normalized angles for edge points
        angle1 = self._get_angle_from_center(self.edge_point1)
        angle2 = self._get_angle_from_center(self.edge_point2)
        
        # Calculate sector angle (clockwise from edge_point1 to edge_point2)
        sector_angle = angle2 - angle1
        if sector_angle < 0:
            sector_angle += 2 * math.pi
        
        # Collect all points that will form triangle vertices
        points = []
        point_angles = []
        
        # Always start with edge_point1
        points.append(self.edge_point1)
        point_angles.append(angle1)
        
        # Get the four image corners
        corners = [
            (0, 0),                              # Top-left
            (self.bound_width, 0),               # Top-right
            (self.bound_width, self.bound_height), # Bottom-right
            (0, self.bound_height)               # Bottom-left
        ]
        
        # Find corners inside the sector and collect them
        corners_in_sector = []
        for corner in corners:
            if self._is_point_in_sector(corner):
                corner_angle = self._get_angle_from_center(corner)
                corners_in_sector.append((corner, corner_angle))
        
        # Sort corners by angle
        corners_in_sector.sort(key=lambda x: x[1])
        
        # Determine how many mid-points we need
        num_corners = len(corners_in_sector)
        
        if num_corners == 2:
            # Case 1: Two corners - use both corners, no mid-points needed
            points.extend([corner for corner, _ in corners_in_sector])
            point_angles.extend([angle for _, angle in corners_in_sector])
        
        elif num_corners == 1:
            # Case 2: One corner - need one mid-point
            corner, corner_angle = corners_in_sector[0]
            
            # Determine which side needs splitting
            # Calculate relative angle from edge_point1
            corner_rel_angle = corner_angle - angle1
            if corner_rel_angle < 0:
                corner_rel_angle += 2 * math.pi
            
            if corner_rel_angle < sector_angle / 2:
                # Corner is closer to edge_point1, split the larger side (corner to edge_point2)
                # Calculate angle halfway between corner and edge_point2
                if corner_angle <= angle2:
                    mid_angle = self._normalize_angle((corner_angle + angle2) / 2)
                else:
                    # Handle wrap-around: corner_angle > angle2
                    # Need to go clockwise: corner_angle -> 2π -> angle2
                    total_angle = (2 * math.pi - corner_angle) + angle2
                    mid_angle = self._normalize_angle(corner_angle + total_angle / 2)
                
                # Add corner and mid-point in order
                points.append(corner)
                point_angles.append(corner_angle)
                
                mid_point = self._get_point_at_angle(mid_angle)
                mid_angle_actual = self._get_angle_from_center(mid_point)
                points.append(mid_point)
                point_angles.append(mid_angle_actual)
            else:
                # Corner is closer to edge_point2, split the larger side (edge_point1 to corner)
                # Calculate angle halfway between edge_point1 and corner
                mid_angle = self._normalize_angle((angle1 + corner_angle) / 2)
                
                mid_point = self._get_point_at_angle(mid_angle)
                mid_angle_actual = self._get_angle_from_center(mid_point)
                points.append(mid_point)
                point_angles.append(mid_angle_actual)
                
                points.append(corner)
                point_angles.append(corner_angle)
        
        else:
            # Case 3: No corners - need two mid-points to divide sector into 3 equal parts
            third_angle = sector_angle / 3
            mid_angle1 = self._normalize_angle(angle1 + third_angle)
            mid_angle2 = self._normalize_angle(angle1 + 2 * third_angle)
            
            mid_point1 = self._get_point_at_angle(mid_angle1)
            mid_point2 = self._get_point_at_angle(mid_angle2)
            
            mid_angle1_actual = self._get_angle_from_center(mid_point1)
            mid_angle2_actual = self._get_angle_from_center(mid_point2)
            
            points.extend([mid_point1, mid_point2])
            point_angles.extend([mid_angle1_actual, mid_angle2_actual])
        
        # Always end with edge_point2
        points.append(self.edge_point2)
        point_angles.append(angle2)
        
        # Sort all points by angle relative to angle1 (clockwise order)
        # Handle wrap-around: if angle < angle1, treat it as angle + 2π for sorting
        point_angle_pairs = list(zip(points, point_angles))
        
        def get_relative_angle(angle):
            """Get angle relative to angle1 for sorting (handles wrap-around)"""
            rel_angle = angle - angle1
            if rel_angle < 0:
                rel_angle += 2 * math.pi
            return rel_angle
        
        # Sort by relative angle from angle1
        point_angle_pairs.sort(key=lambda x: get_relative_angle(x[1]))
        
        # Remove duplicates (points with same angle, using relative angles)
        unique_points = []
        seen_relative_angles = set()
        for point, angle in point_angle_pairs:
            rel_angle = get_relative_angle(angle)
            rel_angle_rounded = round(rel_angle, 8)
            if rel_angle_rounded not in seen_relative_angles:
                unique_points.append(point)
                seen_relative_angles.add(rel_angle_rounded)
        
        # Ensure we have at least 3 points (edge_point1, at least 1 mid-point/corner, edge_point2)
        if len(unique_points) < 3:
            # Fallback: just divide sector into 3 equal parts
            third_angle = sector_angle / 3
            mid_angle1 = self._normalize_angle(angle1 + third_angle)
            mid_angle2 = self._normalize_angle(angle1 + 2 * third_angle)
            unique_points = [
                self.edge_point1,
                self._get_point_at_angle(mid_angle1),
                self._get_point_at_angle(mid_angle2),
                self.edge_point2
            ]
            # Remove duplicates again
            unique_points = list(dict.fromkeys(unique_points))  # Preserves order
        
        # Create triangles: [center, point_i, point_i+1]
        triangles = []
        for i in range(len(unique_points) - 1):
            triangles.append([self.center, unique_points[i], unique_points[i + 1]])
        
        # Should always have exactly 3 triangles
        if len(triangles) == 3:
            return tuple(triangles)
        elif len(triangles) > 3:
            # If we have more than 3, return first 3
            return tuple(triangles[:3])
        else:
            # If we have fewer than 3, pad with degenerate triangles (shouldn't happen)
            while len(triangles) < 3:
                triangles.append([self.center, self.edge_point1, self.edge_point2])
            return tuple(triangles[:3])

    def _is_point_in_sector(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point is inside the sector.
        
        Args:
            point: (x, y) coordinates to check
            
        Returns:
            True if point is inside the sector, False otherwise
        """
        # Vector from center to point
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        
        # Angle to the point
        point_angle = math.atan2(dy, dx)
        
        # Angles to edge points
        angle1 = math.atan2(self.edge_point1[1] - self.center[1], 
                            self.edge_point1[0] - self.center[0])
        angle2 = math.atan2(self.edge_point2[1] - self.center[1], 
                            self.edge_point2[0] - self.center[0])
        
        # Normalize angles to [0, 2π)
        def normalize_angle(a):
            while a < 0:
                a += 2 * math.pi
            while a >= 2 * math.pi:
                a -= 2 * math.pi
            return a
        
        point_angle = normalize_angle(point_angle)
        angle1 = normalize_angle(angle1)
        angle2 = normalize_angle(angle2)
        
        # Check if point_angle is between angle1 and angle2 (clockwise)
        if angle1 <= angle2:
            return angle1 <= point_angle <= angle2
        else:
            return point_angle >= angle1 or point_angle <= angle2

    def _sort_points_clockwise(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Sort points clockwise relative to center.
        
        Args:
            points: List of (x, y) coordinates
            
        Returns:
            Sorted list of points in clockwise order
        """
        def angle_from_center(point):
            return math.atan2(point[1] - self.center[1], 
                            point[0] - self.center[0])
        
        return sorted(points, key=angle_from_center)


    
class ImageSectors:
    def __init__(self, sectors):
        """
        Initialises a sectorized image based on the sectors.
        Args:
            sectors: List of LinearBoundedSector instances
        """
        self.sector_abdominal_wall, self.sector_attached_tissue, self.sector_detached_tissue = sectors

    @staticmethod
    def get_image_sectors_from_sample(sample):
        """
        Creates an ImageSectors instance from a sample dictionary.
        
        Args:
            sample: Sample dictionary containing:
                - "frame": Image (numpy array)
                - "points": List of raw points (torch tensors or lists)
                
        Returns:
            ImageSectors instance with calculated sectors
            
        Example:
            sample = {
                "frame": img,
                "points": points
            }
            image_sectors = ImageSectors.get_image_sectors_from_sample(sample)
        """
        img = sample["frame"]
        # calculate sectors
        height, width = img.shape[:2]
        raw_points = sample["points"]
        sectors = LinearBoundedSector.get_sectors_from_raw_data(
            raw_points=raw_points, 
            bound_height=height, 
            bound_width=width
        )
        
        return ImageSectors(sectors)


    def warp_adhesion_vector(self, angle):
        """
        Turns the adhesion vector in the sectors for the given angle.
        
        The adhesion vector is given by the vector from the center to 
        the edge_point2 of the sector_attached_tissue and
        the edge_point1 of the sector_abdominal_wall

        Function adjusts the sectors of the ImageSector instance accordingly.
        
        Args:
            angle: Angle in degrees to rotate the adhesion vector
                (positive = counterclockwise, negative = clockwise)
        """
        # The adhesion vector is defined by:
        # - edge_point2 of sector_attached_tissue
        # - edge_point1 of sector_abdominal_wall
        # These two points should move together when the adhesion vector rotates
        
        # Rotate edge_point1 of abdominal wall sector
        # (this is the end of the adhesion vector on the abdominal wall side)
        self.sector_abdominal_wall.add_angle_to_sector(angle, start=True)

        # Rotate edge_point2 of attached tissue sector
        # (this is the end of the adhesion vector on the tissue side)
        self.sector_attached_tissue.add_angle_to_sector(angle, start=False)

    def get_radial_triangles(self) -> List[List[Tuple[int, int]]]:
        """
        Returns always 9 radial triangles.
        1. 3 triangles of sector_abdominal_wall
        2. 3 triangles of sector_detached_tissue
        3. 3 triangles of sector_attached_tissue
        
        Returns:
            List of 9 triangles, each triangle is a list of 3 (x, y) coordinates.
            Triangles are ordered clockwise starting from sector_abdominal_wall.
        """
        # Get 3 triangles from each sector
        triangles_abdominal = self.sector_abdominal_wall.get_radial_triangles()
        triangles_detached = self.sector_detached_tissue.get_radial_triangles()
        triangles_attached = self.sector_attached_tissue.get_radial_triangles()
        
        # Combine all triangles in clockwise order
        all_triangles = triangles_abdominal + triangles_detached + triangles_attached
        
        return all_triangles

    

class ImageSectorTransformer:
    def __init__(self):
        pass

    @staticmethod
    def radial_sector_warping(src_image, image_sectors: ImageSectors, angle: int):
        """
        Warps the sectorized_image for the given angle.
        
        Args:
            src_image: Source image to warp (numpy array)
            image_sectors: ImageSectors instance defining the sector structure
            angle: Angle in degrees to rotate the adhesion vector
            
        Returns:
            Warped image (numpy array)
        """
        # Create a deep copy of image_sectors to avoid modifying the original
        warped_sectors = copy.deepcopy(image_sectors)
        
        # Apply the adhesion vector warping to the copied sectors
        warped_sectors.warp_adhesion_vector(angle)
        
        # Get triangles from original sectors (source triangles)
        src_triangles = image_sectors.get_radial_triangles()
        
        # Get triangles from warped sectors (destination triangles)
        dst_triangles = warped_sectors.get_radial_triangles()
        
        # Map all triangles from source to destination
        warped_image = ImageSectorTransformer.map_triangles(src_image, src_triangles, dst_triangles, debug=False)
        
        return warped_image

    @staticmethod
    def sector_mixup(src_image, src_sector: LinearBoundedSector, dst_image, dst_sector: LinearBoundedSector, alpha: float = 0.5):
        """
        Mixes pixels from dst_image in dst_sector into the src_sector in src image.
        
        Args:
            src_image: Source image (will be modified in place and returned)
            src_sector: Source sector defining the target geometry
            dst_image: Destination image to sample pixels from
            dst_sector: Destination sector defining the source geometry
            alpha: Mixing factor [0, 1]. 0 = only src_image, 1 = only warped dst_image
            
        Returns:
            Modified src_image with mixed pixels (numpy array)
        """
        # Get triangles from both sectors
        src_triangles = src_sector.get_radial_triangles()  # Returns tuple of 3 triangles
        dst_triangles = dst_sector.get_radial_triangles()  # Returns tuple of 3 triangles
        
        # Convert tuples to lists for easier iteration
        src_tri_list = list(src_triangles)
        dst_tri_list = list(dst_triangles)
        
        # Create buffer for warped content from dst_image
        warped_buffer = np.zeros_like(src_image, dtype=np.float32)
        
        # Create combined mask for src_sector
        mask_combined = np.zeros(src_image.shape[:2], dtype=np.float32)
        
        # Warp each triangle from dst_image to src_sector geometry
        for dst_tri, src_tri in zip(dst_tri_list, src_tri_list):
            # Warp from dst_image (using dst_tri geometry) to src_tri geometry
            # Note: _warp_triangle_core may raise exceptions for invalid triangles
            try:
                warped_roi, mask, (y1, y2, x1, x2) = ImageSectorTransformer._warp_triangle_core(
                    dst_image, dst_tri, src_tri, src_image.shape[:2]
                )
            except (ValueError, RuntimeError) as e:
                # Skip this triangle if warping fails, but log the error
                import warnings
                warnings.warn(f"Failed to warp triangle in sector_mixup: {e}. Skipping this triangle.")
                continue
            
            # Convert mask to float [0, 1]
            mask_float = mask.astype(np.float32) / 255.0
            mask_3ch_roi = np.dstack([mask_float, mask_float, mask_float])
            
            # Get current buffer ROI
            buffer_roi = warped_buffer[y1:y2, x1:x2]
            
            # Blend warped triangle into buffer (only where mask is active)
            # If multiple triangles overlap, later ones overwrite (sectors shouldn't overlap)
            warped_float = warped_roi.astype(np.float32)
            warped_buffer[y1:y2, x1:x2] = buffer_roi * (1.0 - mask_3ch_roi) + warped_float * mask_3ch_roi
            
            # Update combined mask (use max to combine masks from all triangles)
            mask_combined[y1:y2, x1:x2] = np.maximum(mask_combined[y1:y2, x1:x2], mask_float)
        
        # Create 3-channel mask for blending
        mask_3ch = np.dstack([mask_combined, mask_combined, mask_combined])
        
        # Alpha blend: result = src * (1 - alpha * mask) + warped * (alpha * mask)
        src_float = src_image.astype(np.float32)
        result = src_float * (1.0 - alpha * mask_3ch) + warped_buffer * (alpha * mask_3ch)
        
        # Convert back to uint8 and return
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _warp_triangle_core(image, src_tri, dst_tri, dst_image_shape):
        """
        Core warping logic: warps a triangle and returns warped pixels and mask.
        
        Args:
            image: Source image to warp from (numpy array)
            src_tri: Source triangle [(x1, y1), (x2, y2), (x3, y3)]
            dst_tri: Destination triangle [(x1, y1), (x2, y2), (x3, y3)]
            dst_image_shape: Shape of destination image (height, width) for bounds checking
            
        Returns:
            Tuple of (warped_roi, mask, (y1, y2, x1, x2))
            - warped_roi: Warped triangle pixels (clipped to ROI)
            - mask: Binary mask for the triangle (clipped to ROI, uint8, 0-255)
            - (y1, y2, x1, x2): ROI coordinates in destination image
            
        Raises:
            ValueError: If triangles are invalid, ROIs are invalid, or warping fails
            RuntimeError: If intersection calculation fails
        """
        # Convert to numpy arrays
        src_tri_array = np.array(src_tri, dtype=np.float32)
        dst_tri_array = np.array(dst_tri, dtype=np.float32)
        
        # Get bounding rectangles
        src_rect = cv2.boundingRect(src_tri_array)
        dst_rect = cv2.boundingRect(dst_tri_array)
        
        # Check if rectangles are invalid
        if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            raise ValueError(f"Invalid triangle bounding rectangles: src_rect={src_rect}, dst_rect={dst_rect}")
        
        # Destination ROI clipped to image bounds
        y1_dst, y2_dst = max(0, dst_rect[1]), min(dst_image_shape[0], dst_rect[1] + dst_rect[3])
        x1_dst, x2_dst = max(0, dst_rect[0]), min(dst_image_shape[1], dst_rect[0] + dst_rect[2])
        if y2_dst <= y1_dst or x2_dst <= x1_dst:
            raise ValueError(f"Invalid destination ROI after clipping: ROI=({y1_dst}, {y2_dst}, {x1_dst}, {x2_dst}), image_shape={dst_image_shape}, dst_rect={dst_rect}")
        
        # Offset triangle points by bounding box corner
        src_tri_cropped = src_tri_array - np.array([[src_rect[0], src_rect[1]]], dtype=np.float32)
        dst_tri_cropped = dst_tri_array - np.array([[dst_rect[0], dst_rect[1]]], dtype=np.float32)
        
        # Source crop clipped to image bounds
        y1_src, y2_src = max(0, src_rect[1]), min(image.shape[0], src_rect[1] + src_rect[3])
        x1_src, x2_src = max(0, src_rect[0]), min(image.shape[1], src_rect[0] + src_rect[2])
        if y2_src <= y1_src or x2_src <= x1_src:
            raise ValueError(f"Invalid source ROI after clipping: ROI=({y1_src}, {y2_src}, {x1_src}, {x2_src}), image_shape={image.shape[:2]}, src_rect={src_rect}")
        src_cropped = image[y1_src:y2_src, x1_src:x2_src]
        if src_cropped.size == 0:
            raise ValueError(f"Source crop is empty: ROI=({y1_src}, {y2_src}, {x1_src}, {x2_src}), image_shape={image.shape}")
        
        # Calculate affine transformation
        affine_matrix = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)
        
        # Warp the cropped triangle
        warped = cv2.warpAffine(src_cropped, affine_matrix, 
                            (dst_rect[2], dst_rect[3]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        
        # Create mask for the destination triangle
        mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), 255)
        
        # Clip warped and mask to actual on-image ROI
        h_actual, w_actual = y2_dst - y1_dst, x2_dst - x1_dst
        warped = warped[:h_actual, :w_actual]
        mask = mask[:h_actual, :w_actual]
        if warped.size == 0 or mask.size == 0:
            raise ValueError(f"Warped triangle or mask is empty after clipping: warped_size={warped.size}, mask_size={mask.size}, ROI=({y1_dst}, {y2_dst}, {x1_dst}, {x2_dst})")
        
        return (warped, mask, (y1_dst, y2_dst, x1_dst, x2_dst))

    @staticmethod
    def map_triangle(image, result, src_tri, dst_tri, debug=False, debug_color=None):
        """
        Maps a single source triangle to a destination triangle using affine transformation.
        
        Args:
            image: Input image (numpy array)
            result: Output image to write to (numpy array)
            src_tri: Source triangle [(x1, y1), (x2, y2), (x3, y3)]
            dst_tri: Destination triangle [(x1, y1), (x2, y2), (x3, y3)]
            debug: If True, fill triangle with random color instead of warping image
            debug_color: Optional BGR color tuple for debug mode (if None, uses random color)
        """
        # Convert to numpy arrays
        dst_tri_array = np.array(dst_tri, dtype=np.float32)
        
        # Debug mode: just fill triangle with color
        if debug:
            # Generate random color if not provided
            if debug_color is None:
                debug_color = tuple(np.random.randint(50, 255, 3).tolist())
            
            # Fill the triangle directly on the result image
            cv2.fillPoly(result, [dst_tri_array.astype(np.int32)], debug_color)
            return
        
        # Use core warping method
        # Note: _warp_triangle_core may raise exceptions for invalid triangles
        warped_roi, mask, (y1_dst, y2_dst, x1_dst, x2_dst) = ImageSectorTransformer._warp_triangle_core(
            image, src_tri, dst_tri, result.shape[:2]
        )
        
        # Get the region of interest in result image
        dst_roi = result[y1_dst:y2_dst, x1_dst:x2_dst]
        
        # Apply mask to warped triangle
        warped_masked = cv2.bitwise_and(warped_roi, warped_roi, mask=mask)
        dst_roi_masked = cv2.bitwise_and(dst_roi, dst_roi, mask=cv2.bitwise_not(mask))
        
        # Combine and write back to result image
        result[y1_dst:y2_dst, x1_dst:x2_dst] = cv2.add(dst_roi_masked, warped_masked)

    @staticmethod
    def map_triangles(image, src_triangles, dst_triangles, debug=False):
        """
        Maps source triangles to destination triangles using affine transformations.
        
        Args:
            image: Input image (numpy array)
            src_triangles: List of source triangles, each triangle is [(x1, y1), (x2, y2), (x3, y3)]
            dst_triangles: List of destination triangles, each triangle is [(x1, y1), (x2, y2), (x3, y3)]
            debug: If True, visualize triangles with random colors instead of warping image pixels
            
        Returns:
            Warped image (numpy array) with all triangles mapped
        """
        # Assert same number of triangles
        assert len(src_triangles) == len(dst_triangles), \
            "Source and destination must have the same number of triangles"

        if debug:
            print(f"src_triangles: {src_triangles}")
            print(f"dst_triangles: {dst_triangles}")
        
        # Create output image (black background)
        result = np.zeros_like(image)
        
        # Define 9 deterministic colors for debug visualization (BGR format)
        debug_colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (0, 255, 255),    # Yellow
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 165, 255),    # Orange
            (128, 0, 128),    # Purple
            (147, 20, 255),   # Pink
        ]
        
        # Process each triangle pair
        for idx, (src_tri, dst_tri) in enumerate(zip(src_triangles, dst_triangles)):
            # Use corresponding color from list, or repeat colors if more than 9 triangles
            debug_color = debug_colors[idx % len(debug_colors)] if debug else None
            ImageSectorTransformer.map_triangle(image, result, src_tri, dst_tri, debug=debug, debug_color=debug_color)
        
        return result

    @staticmethod
    def arbitrary_sector_warping(src_image, source_sectors: List[LinearBoundedSector], 
                                  target_sectors: List[LinearBoundedSector], debug: bool = False):
        """
        Warps an image using arbitrary source and target sectors.
        Supports any number of sectors (not limited to 3-sector structure).
        
        Args:
            src_image: Source image to warp (numpy array)
            source_sectors: List of LinearBoundedSector instances defining source geometry
            target_sectors: List of LinearBoundedSector instances defining target geometry
            debug: If True, visualize triangles with colors instead of warping image
            
        Returns:
            Warped image (numpy array)
            
        Raises:
            ValueError: If source and target sectors have different lengths
        """
        if len(source_sectors) != len(target_sectors):
            raise ValueError(f"Source and target must have same number of sectors: "
                           f"source={len(source_sectors)}, target={len(target_sectors)}")
        
        # Collect all triangles from source and target sectors
        src_triangles = []
        dst_triangles = []
        
        for src_sector, dst_sector in zip(source_sectors, target_sectors):
            # Each sector generates 3 triangles
            src_tris = src_sector.get_radial_triangles()
            dst_tris = dst_sector.get_radial_triangles()
            
            # Add triangles to lists
            src_triangles.extend(list(src_tris))
            dst_triangles.extend(list(dst_tris))
        
        # Map all triangles from source to destination
        warped_image = ImageSectorTransformer.map_triangles(
            src_image, src_triangles, dst_triangles, debug=debug
        )
        
        return warped_image

    class Visualizer:
        def __init__(self):
            pass

        def show_images(self, images, titles=None, cols=2, figsize=(12, 8), cmap_gray=False, show=False, return_fig=False):
            """
            Display a gallery of images using matplotlib.

            Parameters
            - images: List of images (BGR or grayscale NumPy arrays).
            - titles: Optional list of titles per image.
            - cols: Number of columns in the grid layout.
            - figsize: Matplotlib figure size (width, height) in inches.
            - cmap_gray: Force grayscale colormap for single-channel images.
            - show: If True, block and show only this figure.
            - return_fig: If True, return the matplotlib Figure (no auto-show).
            """
            n = len(images)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            axes = np.array(axes).reshape(-1)

            for i, ax in enumerate(axes):
                if i < n:
                    img = images[i]
                    # BGR -> RGB wenn Farbbild (3 Kanäle)
                    if img is not None and img.ndim == 3 and img.shape[2] == 3:
                        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(disp)
                    else:
                        ax.imshow(img, cmap='gray' if cmap_gray or (img is not None and img.ndim == 2) else None)
                    ax.axis('off')
                    if titles and i < len(titles):
                        ax.set_title(titles[i])
                else:
                    ax.axis('off')

            plt.tight_layout()
            
            # If return_fig is True, return the figure without displaying it
            if return_fig:
                return fig
            
            # Show only this specific figure when requested (make it current and block)
            if show:
                plt.figure(fig.number)
                plt.show()



if __name__ == "__main__":
    img_path = "./frames/" + "964311118002B0000894" + ".png"
    img_path2 = "./frames/" + "964311118002B0001472" + ".png"
    json_path = "./attachment_graphs/" + "964311118002B0000894" + ".json"
    json_path2 = "./attachment_graphs/" + "964311118002B0001472" + ".json"

    src_image = cv2.imread(img_path)

    src_image2 = cv2.imread(img_path2)
    
    with open(json_path, 'r') as file:
            data = json.load(file)
    points = [torch.tensor([x, y]) for _, (x, y), _ in data]

    with open(json_path2, 'r') as file:
            data2 = json.load(file)
    points2 = [torch.tensor([x, y]) for _, (x, y), _ in data]

    print(points)

    sample = {
        "frame": src_image,
        "points": points
    }

    sample2 = {
        "frame": src_image2,
        "points": points2
    }

    src_sectors = ImageSectors.get_image_sectors_from_sample(sample)
    dst_sectors = ImageSectors.get_image_sectors_from_sample(sample2)

    src_sector = src_sectors.sector_abdominal_wall
    dst_sector = dst_sectors.sector_abdominal_wall

    warped_image1 = ImageSectorTransformer.radial_sector_warping(
        src_image=src_image,
        image_sectors=src_sectors,
        angle=0
    )

    warped_image2 = ImageSectorTransformer.sector_mixup(
        src_image=src_image,
        src_sector=src_sector,
        dst_image=src_image2,
        dst_sector=dst_sector,
        alpha=0.5
    )

    warped_image3 = ImageSectorTransformer.radial_sector_warping(
        src_image=src_image,
        image_sectors=src_sectors,
        angle=-40
    )

    # Display both original and warped images using the show_images method
    visualizer = ImageSectorTransformer.Visualizer()
    visualizer.show_images(
        images=[src_image, warped_image1, warped_image2, warped_image3],
        titles=["Original Image", "Warped Image 0 Grad", "Mixup", "Warped Image -40 Grad"],
        cols=2,
        figsize=(12, 6),
        show=True
    )

