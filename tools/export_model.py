"""Export what the browser demo needs to draw a G1 and the protected person.

Writes into demo/model/:
  g1.json          kinematic tree (bodies, joints, qpos addresses) and visual mesh placements
  g1_meshes.*      decimated Unitree G1 visual meshes (copied from the PAC-MAN site build,
                   which decimates the same 35 STL files kimodo's g1.xml references)
  person.bin/json  the protected person: kimodo's SOMA skin, posed standing, Z-up

Run from the csf repo environment:
  ~/twist2/csf/.venv/bin/python tools/export_model.py
"""
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

SITE = Path(__file__).resolve().parents[1]
OUT = SITE / "demo" / "model"
CSF = Path.home() / "twist2" / "csf"
PACMAN = Path.home() / "twist2" / "perceptive_cbf_rl"
G1_XML = CSF / "third_party/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"

# kimodo (Y up, Z forward) -> MuJoCo (Z up, X forward)
K2M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).T


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rot(q, v):
    return quat_mul(quat_mul(q, np.array([0.0, *v])), quat_conj(q))[1:]


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def export_g1(m):
    bodies = []
    for b in range(1, m.nbody):
        joint = None
        if m.body_jntnum[b] > 0:
            j = m.body_jntadr[b]
            kind = {int(mujoco.mjtJoint.mjJNT_FREE): "free", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}[int(m.jnt_type[j])]
            joint = {"type": kind, "axis": m.jnt_axis[j].tolist(), "pos": m.jnt_pos[j].tolist(),
                     "qposadr": int(m.jnt_qposadr[j])}
        bodies.append({"name": m.body(b).name, "parent": int(m.body_parentid[b]) - 1,
                       "pos": m.body_pos[b].tolist(), "quat": m.body_quat[b].tolist(), "joint": joint})
    # Visual meshes: MuJoCo re-centres each mesh at compile time (mesh_pos, mesh_quat);
    # undo that so the placement applies to the raw STL vertices the web meshes keep.
    geoms = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
            continue
        mesh = m.geom_dataid[g]
        q_inv = quat_conj(m.mesh_quat[mesh])
        quat = quat_mul(m.geom_quat[g], q_inv)
        pos = m.geom_pos[g] - quat_rot(quat, m.mesh_pos[mesh])
        geoms.append({"body": int(m.geom_bodyid[g]) - 1, "mesh": m.mesh(mesh).name,
                      "pos": pos.tolist(), "quat": quat.tolist()})
    return {"nq": int(m.nq), "bodies": bodies, "geoms": geoms}


def forward_kinematics(tree, qpos):
    """The same FK the browser runs (demo/fk.js), for checking against MuJoCo."""
    xpos, xquat = [], []
    for b in tree["bodies"]:
        if b["parent"] < 0:
            ppos, pquat = np.zeros(3), np.array([1.0, 0, 0, 0])
        else:
            ppos, pquat = xpos[b["parent"]], xquat[b["parent"]]
        pos = ppos + quat_rot(pquat, b["pos"])
        quat = quat_mul(pquat, b["quat"])
        j = b["joint"]
        if j and j["type"] == "free":
            a = j["qposadr"]
            pos, quat = qpos[a:a + 3].copy(), qpos[a + 3:a + 7].copy()
        elif j and j["type"] == "hinge":
            r = axis_angle_quat(j["axis"], qpos[j["qposadr"]])
            anchor = pos + quat_rot(quat, j["pos"])
            quat = quat_mul(quat, r)
            pos = anchor - quat_rot(quat, j["pos"])
        xpos.append(pos)
        xquat.append(quat / np.linalg.norm(quat))
    return np.array(xpos), np.array(xquat)


def check_fk(m, tree):
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        q = np.zeros(m.nq)
        q[:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        q[3:7] = quat / np.linalg.norm(quat)
        q[7:] = rng.uniform(-1.0, 1.0, m.nq - 7)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        xpos, xquat = forward_kinematics(tree, q)
        worst = max(worst, np.abs(xpos - d.xpos[1:]).max())
    print(f"FK vs MuJoCo: max body position error {worst:.2e} m")
    assert worst < 1e-6
    # Vectors for the browser FK check (tools/check_fk.mjs).
    vectors = []
    for _ in range(5):
        q = np.zeros(m.nq)
        q[:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        q[3:7] = quat / np.linalg.norm(quat)
        q[7:] = rng.uniform(-1.0, 1.0, m.nq - 7)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        vectors.append({"qpos": q.tolist(), "xpos": d.xpos[1:].tolist(), "xquat": d.xquat[1:].tolist()})
    (SITE / "tools" / "fk_vectors.json").write_text(json.dumps(vectors))


def check_geoms(m, tree):
    """Raw-STL placement must reproduce MuJoCo's compiled mesh vertices."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    worst = 0.0
    import trimesh  # noqa: PLC0415
    name_to_geom = {}
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2:
            name_to_geom.setdefault(m.mesh(m.geom_dataid[g]).name, g)
    mesh_dir = G1_XML.parent.parent / "meshes" / "g1"
    for name, g in list(name_to_geom.items())[:6]:
        stl = trimesh.load(mesh_dir / f"{name}.STL", process=False)
        entry = next(e for e in tree["geoms"] if e["mesh"] == name)
        mid = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        compiled = m.mesh_vert[start:start + count]
        # compiled vertex in geom frame -> body frame
        cq, cp = m.geom_quat[g], m.geom_pos[g]
        body_from_compiled = np.array([quat_rot(cq, v) for v in compiled[:200]]) + cp
        body_from_raw = np.array([quat_rot(entry["quat"], v) for v in stl.vertices]) + entry["pos"]
        # nearest raw vertex for each compiled one
        dist = np.min(np.linalg.norm(body_from_compiled[:, None] - body_from_raw[None], axis=-1), axis=1)
        worst = max(worst, dist.max())
    print(f"mesh placement vs MuJoCo compiled meshes: max error {worst:.2e} m")
    assert worst < 1e-5


def export_person():
    sys.path.insert(0, str(CSF))
    from csf.app.entities import human_mesh  # noqa: PLC0415

    vertices, faces = human_mesh()
    v = (vertices.astype(np.float64) @ K2M.T).astype(np.float32)   # Z-up, facing +X
    (OUT / "person.bin").write_bytes(v.tobytes() + faces.astype(np.uint32).tobytes())
    meta = {"vertexCount": int(len(v)), "indexCount": int(faces.size), "height": float(v[:, 2].max())}
    (OUT / "person.json").write_text(json.dumps(meta))
    print(f"person: {len(v)} vertices, {len(faces)} triangles")


def copy_pacman_meshes():
    for name in ("robot_meshes.bin", "robot_meshes.json"):
        data = subprocess.run(["git", "-C", str(PACMAN), "show", f"gh-pages:demo/model/{name}"],
                              check=True, capture_output=True).stdout
        (OUT / name.replace("robot_meshes", "g1_meshes")).write_bytes(data)
    print("copied the decimated G1 meshes from the PAC-MAN site")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    tree = export_g1(m)
    check_fk(m, tree)
    check_geoms(m, tree)
    meshes = json.loads(subprocess.run(
        ["git", "-C", str(PACMAN), "show", "gh-pages:demo/model/robot_meshes.json"],
        check=True, capture_output=True).stdout)
    available = {e["name"] for e in meshes["meshes"]}
    missing = sorted({g["mesh"] for g in tree["geoms"]} - available)
    assert not missing, f"meshes missing from the web set: {missing}"
    (OUT / "g1.json").write_text(json.dumps(tree))
    copy_pacman_meshes()
    export_person()


if __name__ == "__main__":
    main()
