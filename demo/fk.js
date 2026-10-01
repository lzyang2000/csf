// Forward kinematics of the G1 from MuJoCo qpos, using the tree exported by
// tools/export_model.py (checked there against MuJoCo, and in tools/check_fk.mjs).
// Quaternions are [w, x, y, z]; the world is MuJoCo's: Z up, X forward.

export function quatMul(a, b, out = new Float64Array(4)) {
  const [w1, x1, y1, z1] = a;
  const [w2, x2, y2, z2] = b;
  out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
  out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
  out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
  out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
  return out;
}

export function quatRot(q, v, out = new Float64Array(3)) {
  // v' = q v q*, expanded
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  out[0] = vx + w * tx + (y * tz - z * ty);
  out[1] = vy + w * ty + (z * tx - x * tz);
  out[2] = vz + w * tz + (x * ty - y * tx);
  return out;
}

function axisAngle(axis, angle) {
  const s = Math.sin(angle / 2);
  const n = Math.hypot(axis[0], axis[1], axis[2]);
  return [Math.cos(angle / 2), (s * axis[0]) / n, (s * axis[1]) / n, (s * axis[2]) / n];
}

/**
 * World pose of every body for one qpos.
 * @param tree   g1.json
 * @param qpos   array-like, starting at `offset`
 * @returns {pos: Float64Array(3B), quat: Float64Array(4B)}
 */
export function forwardKinematics(tree, qpos, offset = 0, result = null) {
  const n = tree.bodies.length;
  const pos = result ? result.pos : new Float64Array(3 * n);
  const quat = result ? result.quat : new Float64Array(4 * n);
  const tmp = new Float64Array(3);
  for (let b = 0; b < n; b++) {
    const body = tree.bodies[b];
    let p, q;
    if (body.parent < 0) {
      p = [body.pos[0], body.pos[1], body.pos[2]];
      q = body.quat.slice();
    } else {
      const pp = pos.subarray(3 * body.parent, 3 * body.parent + 3);
      const pq = quat.subarray(4 * body.parent, 4 * body.parent + 4);
      quatRot(pq, body.pos, tmp);
      p = [pp[0] + tmp[0], pp[1] + tmp[1], pp[2] + tmp[2]];
      q = Array.from(quatMul(pq, body.quat));
    }
    const j = body.joint;
    if (j && j.type === 'free') {
      const a = offset + j.qposadr;
      p = [qpos[a], qpos[a + 1], qpos[a + 2]];
      q = [qpos[a + 3], qpos[a + 4], qpos[a + 5], qpos[a + 6]];
    } else if (j && j.type === 'hinge') {
      const r = axisAngle(j.axis, qpos[offset + j.qposadr]);
      const anchor = quatRot(q, j.pos);
      const ax = [p[0] + anchor[0], p[1] + anchor[1], p[2] + anchor[2]];
      q = Array.from(quatMul(q, r));
      const back = quatRot(q, j.pos);
      p = [ax[0] - back[0], ax[1] - back[1], ax[2] - back[2]];
    }
    const norm = Math.hypot(q[0], q[1], q[2], q[3]);
    pos.set(p, 3 * b);
    quat.set([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm], 4 * b);
  }
  return { pos, quat };
}
