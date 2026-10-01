// Check demo/fk.js against MuJoCo body poses written by tools/export_model.py.
//   node tools/check_fk.mjs
import { readFileSync } from 'node:fs';
import { forwardKinematics } from '../demo/fk.js';

const tree = JSON.parse(readFileSync(new URL('../demo/model/g1.json', import.meta.url)));
const vectors = JSON.parse(readFileSync(new URL('./fk_vectors.json', import.meta.url)));
let worstPos = 0;
let worstQuat = 0;
for (const v of vectors) {
  const { pos, quat } = forwardKinematics(tree, v.qpos);
  v.xpos.forEach((p, b) => p.forEach((x, i) => { worstPos = Math.max(worstPos, Math.abs(x - pos[3 * b + i])); }));
  v.xquat.forEach((q, b) => {
    const dot = Math.abs(q.reduce((s, x, i) => s + x * quat[4 * b + i], 0));
    worstQuat = Math.max(worstQuat, 1 - dot);
  });
}
console.log(`fk.js vs MuJoCo: max position error ${worstPos.toExponential(2)} m, max quaternion 1-|dot| ${worstQuat.toExponential(2)}`);
if (worstPos > 1e-9 || worstQuat > 1e-12) process.exit(1);
