#!/usr/bin/python3
"""
Pygame SLAM GUI for F1Tenth
- Shows live SLAM map (/map) with robot pose overlay
- Keyboard driving (WASD / arrow keys)
- Autopilot button (P key or click) — relays /stanley/drive → /drive
"""

import math
import threading
from collections import deque

import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener

# ── colours ──────────────────────────────────────────────────────────────────
C_BG        = ( 30,  30,  30)
C_UNKNOWN   = (100, 100, 100)
C_FREE      = (220, 220, 220)
C_OCCUPIED  = ( 20,  20,  20)
C_ROBOT_REAL = ( 50, 200,  50)   # green  = ground truth (odom)
C_ROBOT_SLAM = ( 80, 140, 255)   # blue   = SLAM estimated (map TF)
C_HEADING    = (255,  80,  80)
C_PANEL     = ( 20,  20,  20)
C_TEXT      = (230, 230, 230)
C_ACCENT    = ( 80, 160, 255)
C_WARN      = (255, 180,  40)
C_AP_ON     = ( 40, 200,  80)   # autopilot active
C_AP_OFF    = ( 80,  80,  80)   # autopilot inactive
C_AP_BORDER = (200, 200, 200)

# ── layout ────────────────────────────────────────────────────────────────────
WIN_W, WIN_H  = 1100, 900
MAP_RECT      = pygame.Rect(0,   0,   900, 580)
PANEL_RECT    = pygame.Rect(900, 0,   200, 900)
GRAPH_YAW     = pygame.Rect(0,   580, 450, 320)   # heading: real vs SLAM
GRAPH_ERR     = pygame.Rect(450, 580, 450, 320)   # displacement error
BTN_RECT      = pygame.Rect(910, 820, 180, 60)

GRAPH_BUF     = 300   # ~10 s at 30 fps

MAX_SPEED    = 2.0
MAX_STEER    = 0.4
SPEED_STEP   = 0.25
STEER_STEP   = 0.05


class SlamGui(Node):
    def __init__(self):
        super().__init__("slam_gui")

        self._lock          = threading.Lock()
        self._map_data      = None
        self._map_info      = None
        # ground truth pose (from sim odom)
        self._robot_x       = 0.0
        self._robot_y       = 0.0
        self._robot_yaw     = 0.0
        # SLAM estimated pose (from map→base_link TF)
        self._slam_x        = None
        self._slam_y        = None
        self._slam_yaw      = None
        self._speed         = 0.0
        self._steer         = 0.0
        self._autopilot     = False
        self._ap_speed      = 0.0
        self._ap_steer      = 0.0
        self._ap_speed_scale = 1.0   # multiplier 0.1 – 2.0

        # ── TF listener for SLAM estimated pose ──────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_timer(0.1, self._update_slam_pose)   # 10 Hz TF lookup

        # ── subscribers ──────────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid,         "/map",              self._map_cb,      1)
        self.create_subscription(Odometry,              "/ego_racecar/odom", self._odom_cb,     1)
        self.create_subscription(AckermannDriveStamped, "/stanley/drive",    self._stanley_cb,  1)

        # ── publisher ────────────────────────────────────────────────────────
        self._drive_pub   = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self._drive_timer = self.create_timer(0.05, self._publish_drive)   # 20 Hz

    # ── ROS callbacks ─────────────────────────────────────────────────────────
    def _map_cb(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        rgb  = np.full((h, w, 3), C_UNKNOWN, dtype=np.uint8)
        rgb[data ==   0] = C_FREE
        rgb[data == 100] = C_OCCUPIED
        with self._lock:
            self._map_data = np.flipud(rgb)
            self._map_info = msg.info

    def _odom_cb(self, msg: Odometry):
        q   = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self._lock:
            self._robot_x   = msg.pose.pose.position.x
            self._robot_y   = msg.pose.pose.position.y
            self._robot_yaw = yaw

    def _stanley_cb(self, msg: AckermannDriveStamped):
        with self._lock:
            self._ap_speed = msg.drive.speed
            self._ap_steer = msg.drive.steering_angle

    def _update_slam_pose(self):
        """Lookup map→base_link from TF — this is the SLAM estimated pose."""
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'ego_racecar/base_link', rclpy.time.Time()
            )
            q   = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            with self._lock:
                self._slam_x   = t.transform.translation.x
                self._slam_y   = t.transform.translation.y
                self._slam_yaw = yaw
        except Exception:
            pass

    def _publish_drive(self):
        msg = AckermannDriveStamped()
        with self._lock:
            if self._autopilot:
                msg.drive.speed          = self._ap_speed * self._ap_speed_scale
                msg.drive.steering_angle = self._ap_steer
            else:
                msg.drive.speed          = self._speed
                msg.drive.steering_angle = self._steer
        self._drive_pub.publish(msg)

    def adjust_ap_speed_scale(self, delta):
        with self._lock:
            self._ap_speed_scale = float(np.clip(self._ap_speed_scale + delta, 0.1, 2.0))

    def get_ap_speed_scale(self):
        with self._lock:
            return self._ap_speed_scale

    # ── control API (called from Pygame thread) ───────────────────────────────
    def set_speed(self, v):
        with self._lock:
            self._speed = float(np.clip(v, -MAX_SPEED, MAX_SPEED))

    def set_steer(self, s):
        with self._lock:
            self._steer = float(np.clip(s, -MAX_STEER, MAX_STEER))

    def get_speed(self):
        with self._lock:
            return self._speed

    def get_steer(self):
        with self._lock:
            return self._steer

    def toggle_autopilot(self):
        with self._lock:
            self._autopilot = not self._autopilot
            if not self._autopilot:   # reset manual speed on exit
                self._speed = 0.0
                self._steer = 0.0
        return self._autopilot

    def get_render_state(self):
        with self._lock:
            ap    = self._autopilot
            spd   = self._ap_speed if ap else self._speed
            steer = self._ap_steer if ap else self._steer
            return (
                self._map_data.copy() if self._map_data is not None else None,
                self._map_info,
                self._robot_x,  self._robot_y,  self._robot_yaw,   # ground truth
                self._slam_x,   self._slam_y,   self._slam_yaw,    # SLAM estimate
                spd, steer, ap,
            )


# ── drawing helpers ───────────────────────────────────────────────────────────
def world_to_pixel(wx, wy, info, surf_w, surf_h, scale):
    ox  = info.origin.position.x
    oy  = info.origin.position.y
    res = info.resolution
    cx  = (wx - ox) / res
    cy  = (wy - oy) / res
    return int(cx * scale), int((surf_h - cy) * scale)


def draw_ground_truth(surf, px, py, yaw, size=11):
    """Ground truth: filled circle with heading line."""
    pygame.draw.circle(surf, C_ROBOT_REAL, (px, py), size)
    pygame.draw.circle(surf, (255, 255, 255), (px, py), size, 2)
    ex = int(px + size * 2.0 * math.cos(yaw))
    ey = int(py - size * 2.0 * math.sin(yaw))
    pygame.draw.line(surf, C_HEADING, (px, py), (ex, ey), 3)


def draw_slam_estimate(surf, px, py, yaw, size=13):
    """SLAM estimate: hollow triangle pointing in heading direction."""
    # Three vertices of an arrow/triangle
    front = (px + size * math.cos(yaw),           py - size * math.sin(yaw))
    left  = (px + size * 0.7 * math.cos(yaw + 2.3), py - size * 0.7 * math.sin(yaw + 2.3))
    right = (px + size * 0.7 * math.cos(yaw - 2.3), py - size * 0.7 * math.sin(yaw - 2.3))
    pts   = [(int(p[0]), int(p[1])) for p in (front, left, right)]
    pygame.draw.polygon(surf, C_ROBOT_SLAM, pts)
    pygame.draw.polygon(surf, (255, 255, 255), pts, 2)   # white outline


def draw_graph(surf, rect, title, font_sm,
               buf_a, label_a, color_a,
               buf_b=None, label_b=None, color_b=None,
               unit=""):
    """Draw a mini rolling time-series graph with up to two traces."""
    C_GRID = (45, 45, 45)
    C_AXIS = (80, 80, 80)
    MX, MY = rect.x + 38, rect.y + 22      # plot origin (top-left of plot area)
    MW, MH = rect.width - 46, rect.height - 36

    pygame.draw.rect(surf, (12, 12, 12), rect)
    pygame.draw.rect(surf, C_AXIS, rect, 1)

    # title
    surf.blit(font_sm.render(title, True, C_TEXT), (rect.x + 5, rect.y + 4))

    all_vals = list(buf_a) + (list(buf_b) if buf_b else [])
    if len(all_vals) < 2:
        msg = font_sm.render("collecting…", True, C_UNKNOWN)
        surf.blit(msg, msg.get_rect(center=rect.center))
        return

    vmin, vmax = min(all_vals), max(all_vals)
    pad = max((vmax - vmin) * 0.1, 0.01)
    vmin -= pad;  vmax += pad
    vrange = vmax - vmin

    # horizontal grid lines + y labels
    N_GRID = 4
    for i in range(N_GRID + 1):
        gy  = MY + int(MH * i / N_GRID)
        val = vmax - vrange * i / N_GRID
        pygame.draw.line(surf, C_GRID, (MX, gy), (MX + MW, gy))
        lbl = font_sm.render(f"{val:.2f}", True, C_AXIS)
        surf.blit(lbl, (rect.x + 1, gy - 7))

    # vertical border lines
    pygame.draw.line(surf, C_AXIS, (MX, MY), (MX, MY + MH))
    pygame.draw.line(surf, C_AXIS, (MX, MY + MH), (MX + MW, MY + MH))

    def to_pts(buf):
        data = list(buf)
        n    = len(data)
        if n < 2:
            return []
        return [
            (MX + int(MW * i / (n - 1)),
             MY + int(MH * (1.0 - (data[i] - vmin) / vrange)))
            for i in range(n)
        ]

    pts_a = to_pts(buf_a)
    if len(pts_a) >= 2:
        pygame.draw.lines(surf, color_a, False, pts_a, 2)

    if buf_b is not None:
        pts_b = to_pts(buf_b)
        if len(pts_b) >= 2:
            pygame.draw.lines(surf, color_b, False, pts_b, 2)

    # legend (bottom-right)
    lx = MX + MW - 120
    ly = MY + MH + 4
    pygame.draw.line(surf, color_a, (lx, ly + 5), (lx + 14, ly + 5), 2)
    surf.blit(font_sm.render(label_a, True, color_a), (lx + 17, ly - 1))
    if buf_b is not None:
        lx2 = lx + 70
        pygame.draw.line(surf, color_b, (lx2, ly + 5), (lx2 + 14, ly + 5), 2)
        surf.blit(font_sm.render(label_b, True, color_b), (lx2 + 17, ly - 1))

    # unit (top-right)
    surf.blit(font_sm.render(unit, True, C_AXIS), (MX + MW - 30, rect.y + 4))


def draw_autopilot_btn(surf, font, rect, active):
    color  = C_AP_ON  if active else C_AP_OFF
    label  = "AUTOPILOT ON"  if active else "AUTOPILOT OFF"
    l_col  = (10, 10, 10)    if active else C_TEXT
    pygame.draw.rect(surf, color,      rect, border_radius=8)
    pygame.draw.rect(surf, C_AP_BORDER, rect, 2, border_radius=8)
    txt = font.render(label, True, l_col)
    surf.blit(txt, txt.get_rect(center=rect.center))


def draw_panel(surf, font_big, font_sm, speed, steer, has_map, autopilot,
               pos_err, yaw_err, ap_scale, rect):
    pygame.draw.rect(surf, C_PANEL, rect)
    x, y = rect.x + 10, rect.y + 15
    lh   = 28

    def txt(text, color=C_TEXT, big=False):
        nonlocal y
        f = font_big if big else font_sm
        surf.blit(f.render(text, True, color), (x, y))
        y += lh + (6 if big else 0)

    txt("F1Tenth SLAM", C_ACCENT, big=True)
    y += 6

    mode_col = C_AP_ON if autopilot else C_WARN
    txt(f"Mode: {'AUTO' if autopilot else 'MANUAL'}", mode_col)
    y += 6

    txt("── Drive ──", C_WARN)
    txt(f"Speed {speed:+.2f} m/s")
    txt(f"Steer {math.degrees(steer):+.1f} °")
    if autopilot:
        sc_col = C_AP_ON if 0.9 <= ap_scale <= 1.1 else C_WARN
        txt(f"AP x{ap_scale:.1f} [/]", sc_col)
    y += 8

    txt("── SLAM Error ──", C_WARN)
    if pos_err is not None:
        # colour: green < 0.1 m, yellow < 0.5 m, red >= 0.5 m
        if pos_err < 0.1:
            ecol = C_ROBOT_REAL
        elif pos_err < 0.5:
            ecol = C_WARN
        else:
            ecol = (220, 60, 60)
        txt(f"Pos  {pos_err:.3f} m", ecol)
        txt(f"Yaw  {math.degrees(yaw_err):+.1f} °", ecol)
    else:
        txt("No SLAM pose yet", C_UNKNOWN)
    y += 8

    txt("── Legend ──", C_WARN)
    # draw mini shapes inline
    pygame.draw.circle(surf, C_ROBOT_REAL, (x + 6, y + 7), 6)
    surf.blit(font_sm.render(" Truth", True, C_ROBOT_REAL), (x + 14, y))
    y += lh
    tri_pts = [(x+6, y+1), (x+1, y+13), (x+11, y+13)]
    pygame.draw.polygon(surf, C_ROBOT_SLAM, tri_pts)
    pygame.draw.polygon(surf, (255, 255, 255), tri_pts, 1)
    surf.blit(font_sm.render(" SLAM", True, C_ROBOT_SLAM), (x + 14, y))
    y += lh + 6

    txt("── Keys ──", C_WARN)
    for line in ["W/↑  faster", "S/↓  slower", "A/←  left",
                 "D/→  right", "Space  stop", "P  autopilot",
                 "[/]  AP speed", "Q  quit"]:
        txt(line)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = SlamGui()

    def _spin():
        try:
            rclpy.spin(node)
        except Exception:
            pass

    threading.Thread(target=_spin, daemon=True).start()

    pygame.init()
    screen   = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("F1Tenth SLAM GUI")
    font_big = pygame.font.SysFont("monospace", 18, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 14)
    font_btn = pygame.font.SysFont("monospace", 13, bold=True)
    clock    = pygame.time.Clock()

    # rolling history buffers
    buf_yaw_err = deque(maxlen=GRAPH_BUF)
    buf_err     = deque(maxlen=GRAPH_BUF)

    running = True
    while running:
        # ── events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    node.set_speed(0.0)
                    node.set_steer(0.0)
                elif event.key == pygame.K_p:
                    node.toggle_autopilot()
                elif event.key == pygame.K_RIGHTBRACKET:
                    node.adjust_ap_speed_scale(+0.1)
                elif event.key == pygame.K_LEFTBRACKET:
                    node.adjust_ap_speed_scale(-0.1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if BTN_RECT.collidepoint(event.pos):
                    node.toggle_autopilot()

        # ── held-key driving (manual mode only) ──────────────────────────────
        *_, autopilot = node.get_render_state()
        if not autopilot:
            keys  = pygame.key.get_pressed()
            spd   = node.get_speed()
            steer = node.get_steer()

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                spd = min(spd + SPEED_STEP * 0.05, MAX_SPEED)
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                spd = max(spd - SPEED_STEP * 0.05, -MAX_SPEED)
            if not (keys[pygame.K_w] or keys[pygame.K_s] or
                    keys[pygame.K_UP] or keys[pygame.K_DOWN]):
                spd *= 0.96

            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                steer = min(steer + STEER_STEP * 0.05, MAX_STEER)
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                steer = max(steer - STEER_STEP * 0.05, -MAX_STEER)
            else:
                steer *= 0.90

            node.set_speed(spd)
            node.set_steer(steer)

        # ── render ───────────────────────────────────────────────────────────
        screen.fill(C_BG)
        map_rgb, info, rx, ry, ryaw, sx, sy, syaw, spd, steer, autopilot = node.get_render_state()

        # map + robots
        if map_rgb is not None:
            h, w    = map_rgb.shape[:2]
            scale   = min(MAP_RECT.width / w, MAP_RECT.height / h)
            sw, sh  = int(w * scale), int(h * scale)
            surf    = pygame.surfarray.make_surface(map_rgb.swapaxes(0, 1))
            surf    = pygame.transform.scale(surf, (sw, sh))
            mox = MAP_RECT.x + (MAP_RECT.width  - sw) // 2
            moy = MAP_RECT.y + (MAP_RECT.height - sh) // 2
            screen.blit(surf, (mox, moy))

            # Ground truth (circle) — drawn first, sits behind SLAM
            px, py = world_to_pixel(rx, ry, info, w, h, scale)
            gpx, gpy = px + mox, py + moy
            draw_ground_truth(screen, gpx, gpy, ryaw)

            # error line + SLAM estimate (triangle) — on top
            if sx is not None:
                spx, spy = world_to_pixel(sx, sy, info, w, h, scale)
                spx += mox;  spy += moy
                pygame.draw.line(screen, (180, 180, 60), (gpx, gpy), (spx, spy), 1)
                draw_slam_estimate(screen, spx, spy, syaw)

            # scale bar
            leg_px = int(1.0 / info.resolution * scale)
            lx2, ly2 = MAP_RECT.x + 10, MAP_RECT.bottom - 15
            pygame.draw.line(screen, C_ACCENT, (lx2, ly2), (lx2 + leg_px, ly2), 3)
            screen.blit(font_sm.render("1 m", True, C_ACCENT), (lx2 + leg_px + 5, ly2 - 7))

            # autopilot indicator
            if autopilot:
                ind = font_big.render("● AUTOPILOT", True, C_AP_ON)
                screen.blit(ind, (MAP_RECT.x + 10, 10))
        else:
            msg = font_big.render("Waiting for /map …", True, C_UNKNOWN)
            screen.blit(msg, msg.get_rect(center=MAP_RECT.center))

        # displacement error + history buffers
        if sx is not None:
            pos_err = math.hypot(rx - sx, ry - sy)
            dyaw    = (ryaw - syaw + math.pi) % (2 * math.pi) - math.pi
            buf_yaw_err.append(math.degrees(dyaw))
            buf_err.append(pos_err)
        else:
            pos_err = None
            dyaw    = None

        # graphs
        draw_graph(screen, GRAPH_YAW, "Heading error", font_sm,
                   buf_yaw_err if buf_yaw_err else deque([0.0]), "Error", C_WARN,
                   unit="deg")
        draw_graph(screen, GRAPH_ERR, "Displacement error", font_sm,
                   buf_err if buf_err else deque([0.0]), "Error", C_WARN,
                   unit="m")

        # panel + button
        draw_panel(screen, font_big, font_sm, spd, steer,
                   map_rgb is not None, autopilot, pos_err, dyaw,
                   node.get_ap_speed_scale(), PANEL_RECT)
        draw_autopilot_btn(screen, font_btn, BTN_RECT, autopilot)

        pygame.display.flip()
        clock.tick(30)

    node.set_speed(0.0)
    node.set_steer(0.0)
    pygame.quit()
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
