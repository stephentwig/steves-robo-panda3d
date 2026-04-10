from __future__ import annotations

from dataclasses import dataclass
from math import sin

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    TextNode,
    Vec3,
    WindowProperties,
)

Size3 = tuple[float, float, float]
Color4 = tuple[float, float, float, float]

MOVE_SPEED = 12.0
JUMP_SPEED = 16.0
GRAVITY = 36.0
PLAYER_WIDTH = 1.2
PLAYER_HEIGHT = 2.4
PLAYER_DEPTH = 1.0
CAMERA_DISTANCE = 28.0


@dataclass(frozen=True)
class Platform:
    node: NodePath
    position: Vec3
    size: Vec3

    @property
    def left(self) -> float:
        return self.position.x - self.size.x / 2.0

    @property
    def right(self) -> float:
        return self.position.x + self.size.x / 2.0

    @property
    def bottom(self) -> float:
        return self.position.z - self.size.z / 2.0

    @property
    def top(self) -> float:
        return self.position.z + self.size.z / 2.0


def make_box(name: str, size: Size3, color: Color4) -> GeomNode:
    """Build a simple lit box centered at the origin."""
    half_x, half_y, half_z = (axis / 2.0 for axis in size)

    faces = [
        (
            (
                (-half_x, -half_y, -half_z),
                (half_x, -half_y, -half_z),
                (half_x, half_y, -half_z),
                (-half_x, half_y, -half_z),
            ),
            (0.0, 0.0, -1.0),
        ),
        (
            (
                (-half_x, -half_y, half_z),
                (-half_x, half_y, half_z),
                (half_x, half_y, half_z),
                (half_x, -half_y, half_z),
            ),
            (0.0, 0.0, 1.0),
        ),
        (
            (
                (-half_x, -half_y, -half_z),
                (-half_x, -half_y, half_z),
                (half_x, -half_y, half_z),
                (half_x, -half_y, -half_z),
            ),
            (0.0, -1.0, 0.0),
        ),
        (
            (
                (-half_x, half_y, -half_z),
                (half_x, half_y, -half_z),
                (half_x, half_y, half_z),
                (-half_x, half_y, half_z),
            ),
            (0.0, 1.0, 0.0),
        ),
        (
            (
                (-half_x, -half_y, -half_z),
                (-half_x, half_y, -half_z),
                (-half_x, half_y, half_z),
                (-half_x, -half_y, half_z),
            ),
            (-1.0, 0.0, 0.0),
        ),
        (
            (
                (half_x, -half_y, -half_z),
                (half_x, -half_y, half_z),
                (half_x, half_y, half_z),
                (half_x, half_y, -half_z),
            ),
            (1.0, 0.0, 0.0),
        ),
    ]

    vertex_data = GeomVertexData(name, GeomVertexFormat.get_v3n3c4(), Geom.UH_static)
    vertices = GeomVertexWriter(vertex_data, "vertex")
    normals = GeomVertexWriter(vertex_data, "normal")
    colors = GeomVertexWriter(vertex_data, "color")
    triangles = GeomTriangles(Geom.UH_static)

    for face_index, (face_vertices, normal) in enumerate(faces):
        vertex_offset = face_index * 4
        for x, y, z in face_vertices:
            vertices.addData3(x, y, z)
            normals.addData3(*normal)
            colors.addData4(*color)

        triangles.addVertices(vertex_offset, vertex_offset + 1, vertex_offset + 2)
        triangles.addVertices(vertex_offset, vertex_offset + 2, vertex_offset + 3)

    geom = Geom(vertex_data)
    geom.addPrimitive(triangles)

    node = GeomNode(name)
    node.addGeom(geom)
    return node


class RoboPandaPlatformer(ShowBase):
    def __init__(self) -> None:
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.60, 0.82, 0.98, 1.0)
        self.accept("escape", self.userExit)

        self.keys = {"left": False, "right": False}
        self.jump_requested = False
        self.platforms: list[Platform] = []
        self.vertical_velocity = 0.0
        self.is_grounded = False
        self.reached_goal = False
        self.facing = 1
        self.goal_flag_base_z = 0.0
        self.status_text: OnscreenText | None = None

        self.start_position = Vec3(-11.0, 0.0, 0.0)
        self.player_position = Vec3(self.start_position)

        if self.win:
            window_props = WindowProperties()
            window_props.setTitle("Steve's Robo Panda Platformer")
            self.win.requestProperties(window_props)

        self._setup_input()
        self._setup_lights()
        self._build_scene()
        self._build_player()
        self._build_ui()
        self.reset_player()

        self.taskMgr.add(self.update_game, "update_game")

    def _setup_input(self) -> None:
        for key_name in ("a", "arrow_left"):
            self.accept(key_name, self._set_key, ["left", True])
            self.accept(f"{key_name}-up", self._set_key, ["left", False])

        for key_name in ("d", "arrow_right"):
            self.accept(key_name, self._set_key, ["right", True])
            self.accept(f"{key_name}-up", self._set_key, ["right", False])

        for key_name in ("space", "w", "arrow_up"):
            self.accept(key_name, self._request_jump)

        self.accept("r", self.reset_player)

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.55, 0.57, 0.62, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor((0.95, 0.90, 0.82, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-25, -35, 0)
        self.render.setLight(sun_np)

    def _build_scene(self) -> None:
        self._add_backdrop(Vec3(3.0, 18.0, 6.0), Vec3(54.0, 0.8, 16.0), (0.80, 0.89, 0.98, 1.0))
        self._add_backdrop(Vec3(-9.0, 12.0, 3.0), Vec3(18.0, 0.8, 6.0), (0.61, 0.76, 0.86, 1.0))
        self._add_backdrop(Vec3(10.0, 12.0, 4.4), Vec3(22.0, 0.8, 8.8), (0.66, 0.79, 0.88, 1.0))

        self._add_platform(Vec3(0.0, 0.0, -1.0), Vec3(30.0, 5.0, 2.0), (0.19, 0.52, 0.24, 1.0))
        self._add_platform(Vec3(-7.0, 0.0, 2.8), Vec3(5.5, 4.0, 0.8), (0.76, 0.49, 0.22, 1.0))
        self._add_platform(Vec3(0.0, 0.0, 4.4), Vec3(4.8, 4.0, 0.8), (0.84, 0.66, 0.25, 1.0))
        self._add_platform(Vec3(6.5, 0.0, 6.8), Vec3(4.2, 4.0, 0.8), (0.66, 0.42, 0.18, 1.0))
        self._add_platform(Vec3(12.5, 0.0, 9.2), Vec3(4.2, 4.0, 0.8), (0.79, 0.54, 0.21, 1.0))
        final_platform = self._add_platform(Vec3(18.5, 0.0, 11.6), Vec3(4.0, 4.0, 0.8), (0.89, 0.78, 0.34, 1.0))

        self.goal_position = Vec3(final_platform.position.x, 0.0, final_platform.top)
        self.goal_size = Vec3(1.6, 0.8, 2.8)

        goal_pole = self.render.attachNewNode(make_box("goal_pole", (0.24, 0.24, 3.0), (0.95, 0.95, 0.98, 1.0)))
        goal_pole.setPos(self.goal_position.x, 0.0, self.goal_position.z + 1.5)

        self.goal_flag = self.render.attachNewNode(
            make_box("goal_flag", (1.2, 0.18, 0.75), (0.95, 0.20, 0.16, 1.0))
        )
        self.goal_flag_base_z = self.goal_position.z + 2.4
        self.goal_flag.setPos(self.goal_position.x + 0.7, -0.15, self.goal_flag_base_z)

    def _build_player(self) -> None:
        self.player_root = self.render.attachNewNode("player")

        body = self.player_root.attachNewNode(make_box("body", (1.2, 0.9, 1.15), (0.93, 0.94, 0.97, 1.0)))
        body.setPos(0.0, 0.0, 0.95)

        belly = self.player_root.attachNewNode(make_box("belly", (0.72, 0.95, 0.8), (0.16, 0.18, 0.22, 1.0)))
        belly.setPos(0.0, -0.02, 0.9)

        head = self.player_root.attachNewNode(make_box("head", (1.0, 0.95, 0.85), (0.95, 0.96, 0.98, 1.0)))
        head.setPos(0.0, 0.0, 2.0)

        left_ear = self.player_root.attachNewNode(make_box("left_ear", (0.28, 0.3, 0.4), (0.12, 0.14, 0.17, 1.0)))
        left_ear.setPos(-0.32, 0.0, 2.55)

        right_ear = self.player_root.attachNewNode(make_box("right_ear", (0.28, 0.3, 0.4), (0.12, 0.14, 0.17, 1.0)))
        right_ear.setPos(0.32, 0.0, 2.55)

        left_eye = self.player_root.attachNewNode(make_box("left_eye", (0.18, 0.08, 0.18), (0.10, 0.12, 0.14, 1.0)))
        left_eye.setPos(-0.18, -0.48, 2.08)

        right_eye = self.player_root.attachNewNode(make_box("right_eye", (0.18, 0.08, 0.18), (0.10, 0.12, 0.14, 1.0)))
        right_eye.setPos(0.18, -0.48, 2.08)

        left_foot = self.player_root.attachNewNode(make_box("left_foot", (0.36, 0.7, 0.22), (0.91, 0.56, 0.22, 1.0)))
        left_foot.setPos(-0.26, 0.0, 0.11)

        right_foot = self.player_root.attachNewNode(make_box("right_foot", (0.36, 0.7, 0.22), (0.91, 0.56, 0.22, 1.0)))
        right_foot.setPos(0.26, 0.0, 0.11)

    def _build_ui(self) -> None:
        if not self.win:
            return

        OnscreenText(
            text="A / D or Arrow Keys to move   Space to jump   R to restart",
            parent=self.aspect2d,
            align=TextNode.ACenter,
            pos=(0.0, 0.90),
            scale=0.06,
            fg=(0.10, 0.16, 0.22, 1.0),
        )

        self.status_text = OnscreenText(
            text="Climb to the red flag.",
            parent=self.aspect2d,
            align=TextNode.ACenter,
            pos=(0.0, 0.80),
            scale=0.07,
            fg=(0.15, 0.17, 0.23, 1.0),
        )

    def _add_backdrop(self, position: Vec3, size: Vec3, color: Color4) -> NodePath:
        backdrop = self.render.attachNewNode(make_box("backdrop", (size.x, size.y, size.z), color))
        backdrop.setPos(position)
        return backdrop

    def _add_platform(self, position: Vec3, size: Vec3, color: Color4) -> Platform:
        node = self.render.attachNewNode(make_box("platform", (size.x, size.y, size.z), color))
        node.setPos(position)
        platform = Platform(node=node, position=Vec3(position), size=Vec3(size))
        self.platforms.append(platform)
        return platform

    def _set_key(self, key: str, value: bool) -> None:
        self.keys[key] = value

    def _request_jump(self) -> None:
        self.jump_requested = True

    def reset_player(self) -> None:
        self.player_position = Vec3(self.start_position)
        self.vertical_velocity = 0.0
        self.is_grounded = True
        self.reached_goal = False
        self.jump_requested = False
        self.facing = 1
        self._sync_player_node()
        self._set_status("Climb to the red flag.")

        if self.goal_flag:
            self.goal_flag.setColorScale(1.0, 1.0, 1.0, 1.0)

        if self.camera:
            self.camera.setPos(self.player_position.x, -CAMERA_DISTANCE, self.player_position.z + 8.5)
            self.camera.lookAt(self.player_position.x, 0.0, self.player_position.z + 2.0)

    def update_game(self, task: Task) -> int:
        dt = min(globalClock.getDt(), 1.0 / 30.0)
        direction = int(self.keys["right"]) - int(self.keys["left"])

        if direction:
            self.facing = direction

        if not self.reached_goal:
            if self.jump_requested and self.is_grounded:
                self.vertical_velocity = JUMP_SPEED
                self.is_grounded = False

            self.jump_requested = False

            self._move_horizontal(direction * MOVE_SPEED * dt)
            self.vertical_velocity -= GRAVITY * dt
            self._move_vertical(self.vertical_velocity * dt)

            if self.player_position.z < -8.0:
                self.reset_player()
                self._set_status("A fall resets the run. Try again.")
            elif self._touching_goal():
                self.reached_goal = True
                self.vertical_velocity = 0.0
                self._set_status("Goal reached! Press R to play again.")
                self.goal_flag.setColorScale(1.1, 1.15, 0.75, 1.0)
        else:
            self.jump_requested = False

        self._sync_player_node()
        self._update_camera(dt)
        self._animate_goal(task.time)
        return Task.cont

    def _move_horizontal(self, dx: float) -> None:
        if abs(dx) < 1e-6:
            return

        current_x = self.player_position.x
        next_x = current_x + dx
        old_left = current_x - PLAYER_WIDTH / 2.0
        old_right = current_x + PLAYER_WIDTH / 2.0
        player_bottom = self.player_position.z
        player_top = self.player_position.z + PLAYER_HEIGHT

        for platform in self.platforms:
            if player_top <= platform.bottom or player_bottom >= platform.top:
                continue

            if dx > 0.0 and old_right <= platform.left and next_x + PLAYER_WIDTH / 2.0 > platform.left:
                next_x = min(next_x, platform.left - PLAYER_WIDTH / 2.0)
            elif dx < 0.0 and old_left >= platform.right and next_x - PLAYER_WIDTH / 2.0 < platform.right:
                next_x = max(next_x, platform.right + PLAYER_WIDTH / 2.0)

        self.player_position.x = next_x

    def _move_vertical(self, dz: float) -> None:
        current_bottom = self.player_position.z
        current_top = current_bottom + PLAYER_HEIGHT
        next_bottom = current_bottom + dz
        landed = False

        if dz <= 0.0:
            landing_height: float | None = None
            for platform in self.platforms:
                if not self._ranges_overlap(
                    self.player_position.x - PLAYER_WIDTH / 2.0,
                    self.player_position.x + PLAYER_WIDTH / 2.0,
                    platform.left,
                    platform.right,
                ):
                    continue

                if current_bottom >= platform.top and next_bottom < platform.top:
                    if landing_height is None or platform.top > landing_height:
                        landing_height = platform.top

            if landing_height is not None:
                next_bottom = landing_height
                self.vertical_velocity = 0.0
                landed = True
        else:
            ceiling_height: float | None = None
            next_top = next_bottom + PLAYER_HEIGHT
            for platform in self.platforms:
                if not self._ranges_overlap(
                    self.player_position.x - PLAYER_WIDTH / 2.0,
                    self.player_position.x + PLAYER_WIDTH / 2.0,
                    platform.left,
                    platform.right,
                ):
                    continue

                if current_top <= platform.bottom and next_top > platform.bottom:
                    if ceiling_height is None or platform.bottom < ceiling_height:
                        ceiling_height = platform.bottom

            if ceiling_height is not None:
                next_bottom = ceiling_height - PLAYER_HEIGHT
                self.vertical_velocity = 0.0

        self.player_position.z = next_bottom
        self.is_grounded = landed

    def _touching_goal(self) -> bool:
        goal_left = self.goal_position.x - self.goal_size.x / 2.0
        goal_right = self.goal_position.x + self.goal_size.x / 2.0
        goal_bottom = self.goal_position.z
        goal_top = self.goal_position.z + self.goal_size.z

        player_left = self.player_position.x - PLAYER_WIDTH / 2.0
        player_right = self.player_position.x + PLAYER_WIDTH / 2.0
        player_bottom = self.player_position.z
        player_top = self.player_position.z + PLAYER_HEIGHT

        return self._ranges_overlap(player_left, player_right, goal_left, goal_right) and self._ranges_overlap(
            player_bottom,
            player_top,
            goal_bottom,
            goal_top,
        )

    def _sync_player_node(self) -> None:
        self.player_root.setPos(self.player_position.x, 0.0, self.player_position.z)
        self.player_root.setH(0.0 if self.facing >= 0 else 180.0)

    def _update_camera(self, dt: float) -> None:
        if not self.camera:
            return

        target = Vec3(self.player_position.x + 2.0, -CAMERA_DISTANCE, self.player_position.z + 8.5)
        current = self.camera.getPos()
        smoothing = min(dt * 5.0, 1.0)
        self.camera.setPos(current + (target - current) * smoothing)
        self.camera.lookAt(self.player_position.x + 2.0, 0.0, self.player_position.z + 1.8)

    def _animate_goal(self, time_value: float) -> None:
        flag_bob = sin(time_value * 3.0) * 0.12
        self.goal_flag.setZ(self.goal_flag_base_z + flag_bob)

    def _set_status(self, text: str) -> None:
        if self.status_text:
            self.status_text.setText(text)

    @staticmethod
    def _ranges_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
        return min(end_a, end_b) > max(start_a, start_b) + 0.02


if __name__ == "__main__":
    app = RoboPandaPlatformer()
    app.run()
