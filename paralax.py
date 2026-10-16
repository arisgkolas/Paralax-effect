import pygame
import cv2
import numpy as np
import threading
import time

# mediapipe removed the legacy `mp.solutions` API in newer releases.
# Try to use it; if it is missing, fall back to OpenCV's built-in face detector
# so the program works either way.
try:
    import mediapipe as mp
    _FACE_MESH_MODULE = mp.solutions.face_mesh
except (ImportError, AttributeError):
    _FACE_MESH_MODULE = None

WIDTH, HEIGHT = 1000, 700
UNIT_SCALE = 300
EYE_DEPTH = 2.0
ROOM_DEPTH = 8.0

NEON_BLUE = (0, 255, 255)
RED = (255, 0, 0)
DARK_RED = (100, 0, 0)
BLACK = (10, 10, 10)  # Slightly off-black for atmosphere


class Point3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def project_off_axis(p: Point3D, hx, hy):
    """
    Projects a 3D point onto 2D screen based on head position (hx, hy).
    hx, hy are normalized (-1 to 1).
    """
    total_depth = EYE_DEPTH + p.z

    # Avoid division by zero or objects behind the camera
    if total_depth <= 0.1:
        return None

    ratio = EYE_DEPTH / total_depth

    # Parallax formula
    screen_x = hx + (p.x - hx) * ratio
    screen_y = hy + (p.y - hy) * ratio

    # Convert to pixel coordinates
    px = int(WIDTH / 2 + screen_x * UNIT_SCALE)
    py = int(HEIGHT / 2 + screen_y * UNIT_SCALE)
    return (px, py)


def rotate3d(p: Point3D, ax, ay, az):
    """Rotates a point around X, Y, Z axes."""
    x, y, z = p.x, p.y, p.z

    # Rotate X
    cosx, sinx = np.cos(ax), np.sin(ax)
    y2 = y * cosx - z * sinx
    z2 = y * sinx + z * cosx
    y, z = y2, z2

    # Rotate Y
    cosy, siny = np.cos(ay), np.sin(ay)
    x2 = x * cosy + z * siny
    z2 = -x * siny + z * cosy
    x, z = x2, z2

    # Rotate Z
    cosz, sinz = np.cos(az), np.sin(az)
    x2 = x * cosz - y * sinz
    y2 = x * sinz + y * cosz
    x, y = x2, y2

    return Point3D(x, y, z)


def draw_room_grid(surface, hx, hy):
    """Draws a 3D wireframe room to give perspective context."""
    w = 3.0  # Room half-width
    h = 2.0  # Room half-height

    # 1. Draw Longitudinal Lines (Depth lines)
    # Floor and Ceiling
    for x in np.arange(-w, w + 0.1, 1.0):
        p_start = project_off_axis(Point3D(x, h, 0), hx, hy)
        p_end = project_off_axis(Point3D(x, h, ROOM_DEPTH), hx, hy)
        if p_start and p_end:
            pygame.draw.line(surface, NEON_BLUE, p_start, p_end, 1)

        p_start = project_off_axis(Point3D(x, -h, 0), hx, hy)
        p_end = project_off_axis(Point3D(x, -h, ROOM_DEPTH), hx, hy)
        if p_start and p_end:
            pygame.draw.line(surface, NEON_BLUE, p_start, p_end, 1)

    # Walls
    for y in np.arange(-h, h + 0.1, 1.0):
        p_start = project_off_axis(Point3D(-w, y, 0), hx, hy)
        p_end = project_off_axis(Point3D(-w, y, ROOM_DEPTH), hx, hy)
        if p_start and p_end:
            pygame.draw.line(surface, NEON_BLUE, p_start, p_end, 1)

        p_start = project_off_axis(Point3D(w, y, 0), hx, hy)
        p_end = project_off_axis(Point3D(w, y, ROOM_DEPTH), hx, hy)
        if p_start and p_end:
            pygame.draw.line(surface, NEON_BLUE, p_start, p_end, 1)

    # 2. Draw Transverse Lines (Slices/Frames)
    for z in np.arange(0, ROOM_DEPTH + 0.1, 1.0):
        tl = project_off_axis(Point3D(-w, -h, z), hx, hy)
        tr = project_off_axis(Point3D(w, -h, z), hx, hy)
        br = project_off_axis(Point3D(w, h, z), hx, hy)
        bl = project_off_axis(Point3D(-w, h, z), hx, hy)

        if tl and tr and br and bl:
            pygame.draw.line(surface, NEON_BLUE, tl, tr, 1)
            pygame.draw.line(surface, NEON_BLUE, tr, br, 1)
            pygame.draw.line(surface, NEON_BLUE, br, bl, 1)
            pygame.draw.line(surface, NEON_BLUE, bl, tl, 1)


def draw_filled_cube(surface, hx, hy, angle):
    """Draws a rotating cube using Painter's Algorithm (sort by depth)."""
    size = 0.5
    cx, cy, cz = 0, 0, 4.0  # Center of the cube in the room

    base_points = [
        Point3D(-size, -size, -size), Point3D(size, -size, -size),
        Point3D(size, size, -size),   Point3D(-size, size, -size),
        Point3D(-size, -size, size),  Point3D(size, -size, size),
        Point3D(size, size, size),    Point3D(-size, size, size),
    ]

    # Rotate points
    rotated_points = [rotate3d(p, angle, angle * 0.5, angle * 0.2) for p in base_points]

    # Translate points to room location
    world_points = [Point3D(p.x + cx, p.y + cy, p.z + cz) for p in rotated_points]

    # Faces (indices of vertices)
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (2, 3, 7, 6),
        (1, 2, 6, 5), (3, 0, 4, 7)
    ]

    # Average depth (Z) per face for sorting
    face_depths = []
    for face_indices in faces:
        avg_z = sum(world_points[i].z for i in face_indices) / 4
        face_depths.append((avg_z, face_indices))

    # Farthest to nearest (Painter's Algorithm)
    face_depths.sort(key=lambda f: f[0], reverse=True)

    for _, indices in face_depths:
        poly_points = []
        for i in indices:
            p2d = project_off_axis(world_points[i], hx, hy)
            if p2d:
                poly_points.append(p2d)

        if len(poly_points) == 4:
            pygame.draw.polygon(surface, DARK_RED, poly_points)  # Fill
            pygame.draw.polygon(surface, RED, poly_points, 3)    # Outline


class HeadTracking:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("WARNING: Could not open webcam. Scene will stay centered.")

        self.head_x = 0.0
        self.head_y = 0.0
        self.running = True
        self.smooth = 0.35  # 0-1: lower = smoother but laggier

        self.face_mesh = None
        self.face_cascade = None

        if _FACE_MESH_MODULE is not None:
            self.backend = "mediapipe"
            self.face_mesh = _FACE_MESH_MODULE.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.backend = "opencv"
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.face_cascade.empty():
                print("WARNING: Could not load OpenCV face cascade.")

        print(f"Head tracking backend: {self.backend}")

        # Start tracking thread
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _set_target(self, nx, ny):
        """nx, ny are normalized 0..1 image coordinates. Smooth into head_x/head_y."""
        target_x = (nx - 0.5) * 2.0
        target_y = (ny - 0.5) * 2.0
        self.head_x += (target_x - self.head_x) * self.smooth
        self.head_y += (target_y - self.head_y) * self.smooth

    def _update_loop(self):
        while self.running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                time.sleep(0.1)
                continue

            # Flip horizontally for mirror view
            image = cv2.flip(image, 1)

            if self.backend == "mediapipe":
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    # Landmark 168 sits between the eyes
                    landmark = results.multi_face_landmarks[0].landmark[168]
                    self._set_target(landmark.x, landmark.y)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)  # faster detection

                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40)
                )

                if len(faces) > 0:
                    # Use the largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    fh, fw = gray.shape
                    cx = (x + w / 2) / fw
                    cy = (y + h * 0.4) / fh  # roughly eye level
                    self._set_target(cx, cy)

            time.sleep(0.01)  # Small sleep to save CPU

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Off-Axis Head Tracking Illusion")
    clock = pygame.time.Clock()

    # Initialize Tracker
    print("Starting Webcam...")
    tracker = HeadTracking()

    # Animation state
    angle = 0.0
    running = True

    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Update Logic
        hx = tracker.head_x * 1.5  # Exaggerate movement sensitivity
        hy = tracker.head_y * 1.5
        angle += 0.02              # Rotation speed

        # Draw
        screen.fill(BLACK)

        # Background grid
        draw_room_grid(screen, hx, hy)

        # Floating object
        draw_filled_cube(screen, hx, hy, angle)

        pygame.display.flip()
        clock.tick(60)

    # Cleanup
    tracker.stop()
    pygame.quit()
    cv2.destroyAllWindows()
    print("Program ended.")


if __name__ == "__main__":
    main()