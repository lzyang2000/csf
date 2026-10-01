// CSF interactive demo: Kimodo's estimate at each sampling step, without and with CSF.
// Everything shown was recorded offline by tools/record_scenes.py from the csf package.

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { forwardKinematics } from './fk.js';

const COLORS = { before: 0xff5a2c, after: 0x1f8fff, person: 0x45c25a };
const STROBE_POSES = 7;
const STEP_SECONDS = 0.9;

const $ = (id) => document.getElementById(id);

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function fetchBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.arrayBuffer();
}

/** One posable G1: a group per body, the decimated meshes hanging off them. */
class Robot {
  constructor(scene, tree, geometries, color, opacity = 1) {
    this.tree = tree;
    this.root = new THREE.Group();
    scene.add(this.root);
    this.material = new THREE.MeshStandardMaterial({
      color, roughness: 0.55, metalness: 0.05,
      transparent: opacity < 1, opacity, depthWrite: opacity >= 1,
    });
    this.bodies = tree.bodies.map(() => {
      const g = new THREE.Group();
      this.root.add(g);
      return g;
    });
    for (const geom of tree.geoms) {
      const geometry = geometries.get(geom.mesh);
      if (!geometry) continue;
      const mesh = new THREE.Mesh(geometry, this.material);
      mesh.position.fromArray(geom.pos);
      mesh.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
      mesh.castShadow = opacity >= 1;
      this.bodies[geom.body].add(mesh);
    }
    this.fk = { pos: new Float64Array(3 * tree.bodies.length), quat: new Float64Array(4 * tree.bodies.length) };
  }

  pose(qpos, offset) {
    forwardKinematics(this.tree, qpos, offset, this.fk);
    const { pos, quat } = this.fk;
    this.bodies.forEach((g, b) => {
      g.position.set(pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]);
      g.quaternion.set(quat[4 * b + 1], quat[4 * b + 2], quat[4 * b + 3], quat[4 * b]);
    });
  }

  set visible(v) { this.root.visible = v; }
  set offsetY(y) { this.root.position.y = y; }
}

class Demo {
  constructor() {
    this.state = { scene: 0, person: true, view: 'motion', step: -1, playingSteps: false };
    this.cache = new Map();
    this.time = 0;
    this.stepClock = 0;
  }

  async init() {
    const [meta, tree, meshManifest, meshBin, personMeta, personBin] = await Promise.all([
      fetchJSON('data/scenes.json'), fetchJSON('model/g1.json'), fetchJSON('model/g1_meshes.json'),
      fetchBuffer('model/g1_meshes.bin'), fetchJSON('model/person.json'), fetchBuffer('model/person.bin'),
    ]);
    this.meta = meta;
    this.tree = tree;
    this.initScene();
    const geometries = this.loadMeshes(meshManifest, meshBin);
    this.loadPerson(personMeta, personBin);

    this.robots = {};
    for (const slot of ['left', 'right']) {
      const color = slot === 'left' ? COLORS.before : COLORS.after;
      const main = new Robot(this.scene, tree, geometries, color, 1);
      const ghosts = [];
      for (let i = 0; i < STROBE_POSES - 1; i++) {
        const opacity = 0.12 + 0.5 * (i / (STROBE_POSES - 2));
        const g = new Robot(this.scene, tree, geometries, color, opacity);
        g.visible = false;
        ghosts.push(g);
      }
      this.robots[slot] = { main, ghosts };
    }
    this.initControls();
    await this.selectScene(0);
    this.renderer.setAnimationLoop((t) => this.frame(t));
  }

  initScene() {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    const canvas = $('view');
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf3f4f6);
    // Side view: the motion runs left to right toward the person. One camera
    // renders two rows (before / after the filter) through scissored viewports.
    this.camera = new THREE.PerspectiveCamera(24, 1, 0.05, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1.4, -5.0, 1.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(1.4, 0, 0.82);
    this.controls.enableDamping = true;
    this.controls.update();

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bcc4, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-2, -3, 6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5, near: 0.5, far: 20 });
    this.scene.add(sun);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40),
      new THREE.MeshStandardMaterial({ color: 0xe6e8ec, roughness: 1 }));
    floor.receiveShadow = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(40, 80, 0xc9ccd2, 0xd6d9de);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.001;
    this.scene.add(grid);

    const resize = () => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      this.renderer.setSize(w, h, false);
      this.size = { w, h };
    };
    new ResizeObserver(resize).observe(canvas);
    resize();
  }

  loadMeshes(manifest, buffer) {
    const positions = new Float32Array(buffer, 0, manifest.positionBytes / 4);
    const indices = new Uint32Array(buffer, manifest.positionBytes);
    const geometries = new Map();
    for (const m of manifest.meshes) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(
        positions.subarray(m.vertexOffset * 3, (m.vertexOffset + m.vertexCount) * 3), 3));
      geo.setIndex(new THREE.BufferAttribute(indices.subarray(m.indexOffset, m.indexOffset + m.indexCount), 1));
      geo.computeVertexNormals();
      geometries.set(m.name, geo);
    }
    return geometries;
  }

  loadPerson(meta, buffer) {
    const positions = new Float32Array(buffer, 0, meta.vertexCount * 3);
    const indices = new Uint32Array(buffer, meta.vertexCount * 12, meta.indexCount);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({ color: COLORS.person, roughness: 0.7 });
    this.person = new THREE.Mesh(geo, material);
    this.person.castShadow = true;
    this.scene.add(this.person);
  }

  initControls() {
    const select = $('scene');
    this.meta.scenes.forEach((s, i) => select.add(new Option(s.label, i)));
    select.addEventListener('change', () => this.selectScene(Number(select.value)));
    $('person').addEventListener('change', (e) => this.selectVariant(e.target.checked));
    for (const btn of document.querySelectorAll('[data-view]')) {
      btn.addEventListener('click', () => { this.state.view = btn.dataset.view; this.syncButtons(); });
    }
    const bar = $('step');
    bar.max = this.meta.keptSteps.length - 1;
    // Open on the last step: the two finished clips.
    if (this.state.step < 0) this.state.step = this.meta.keptSteps.length - 1;
    // While the bar is held, the clip holds its frame, so only the step changes.
    bar.addEventListener('pointerdown', () => { this.scrubbing = true; });
    window.addEventListener('pointerup', () => { this.scrubbing = false; });
    window.addEventListener('pointercancel', () => { this.scrubbing = false; });
    bar.addEventListener('input', () => { this.state.step = Number(bar.value); this.state.playingSteps = false; this.update(); });
    $('play').addEventListener('click', () => {
      if (!this.state.playingSteps && this.state.step >= this.meta.keptSteps.length - 1) this.state.step = 0;
      this.state.playingSteps = !this.state.playingSteps;
      this.stepClock = 0;
      this.update();
    });
    this.syncButtons();
  }

  syncButtons() {
    for (const btn of document.querySelectorAll('[data-view]')) btn.classList.toggle('on', btn.dataset.view === this.state.view);
  }

  async variantData(scene, key) {
    const id = `${scene.id}_${key}`;
    if (!this.cache.has(id)) {
      const buffer = await fetchBuffer(`data/${scene.variants[key].file}`);
      const half = new Uint16Array(buffer);
      const data = new Float32Array(half.length);
      for (let i = 0; i < half.length; i++) data[i] = halfToFloat(half[i]);
      this.cache.set(id, data);
    }
    return this.cache.get(id);
  }

  async selectScene(index) {
    this.state.scene = index;
    $('scene').value = String(index);
    await this.selectVariant(this.state.person);
  }

  async selectVariant(person) {
    this.state.person = person;
    $('person').checked = person;
    const scene = this.meta.scenes[this.state.scene];
    const key = person ? 'person' : 'empty';
    this.variant = scene.variants[key];
    this.data = await this.variantData(scene, key);
    const { pos, facing } = this.variant.person;
    this.person.position.set(pos[0], pos[1], 0);
    this.person.rotation.set(0, 0, Math.atan2(facing[1], facing[0]));
    this.person.visible = person;
    $('prompt').textContent = `“${scene.prompt}”`;
    this.update();
  }

  /** Offset (in floats) of the clip at kept step `s`: which = 0 without CSF, 1 with CSF. */
  clipOffset(which, s) {
    const { frames, nq, keptSteps } = this.meta;
    return (which * keptSteps.length + s) * frames * nq;
  }

  update() {
    const { keptSteps, steps } = this.meta;
    const s = this.state.step;
    $('step').value = String(s);
    $('play').textContent = this.state.playingSteps ? 'Pause' : 'Play steps';
    $('steplabel').textContent = `Step ${keptSteps[s] + 1} of ${steps}`;
    this.renderMargins();
  }

  renderMargins() {
    const box = $('margins');
    const v = this.variant;
    if (!v.activeRules.length) {
      box.innerHTML = '<p class="note">No rule is active in this scene, so the filter leaves every estimate unchanged.</p>';
      return;
    }
    const s = this.state.step;
    const before = v.marginsBefore[s];
    const after = v.marginsAfter[s];
    const scale = Math.max(1e-9, ...before.map(Math.abs), ...after.map(Math.abs));
    const x = (h) => 8 + 76 * (Math.max(-1, Math.min(0.3, h / scale)) + 1) / 1.3;
    const fmt = (h) => {
      const t = Math.abs(h) >= 10 ? h.toFixed(0) : h.toFixed(2);
      return Number(t) === 0 ? t.replace('-', '') : t;
    };
    const why = v.signals.includes('prompt') ? 'the command' : 'the person in view';
    let html = `<p class="note">Rules active because of ${why}. Margin h of each rule before → after CSF at this step (h ≥ 0 is safe). Safe reference: “${v.safeReference}”.</p>`;
    v.activeRules.forEach((name, r) => {
      html += `<div class="rule"><span class="name">${name}</span>
        <span class="track"><span class="zero" style="left:${x(0)}%"></span>
        <span class="dot before" style="left:${x(before[r])}%"></span>
        <span class="dot after" style="left:${x(after[r])}%"></span></span>
        <span class="val">h ${fmt(before[r])} → ${fmt(after[r])}</span></div>`;
    });
    box.innerHTML = html;
  }

  pose(slot, which, s, frame) {
    const off = this.clipOffset(which, s) + frame * this.meta.nq;
    this.robots[slot].main.pose(this.data, off);
  }

  frame(tMs) {
    const now = tMs / 1000;
    const dt = this.last === undefined ? 0 : Math.min(0.1, now - this.last);
    this.last = now;
    if (!this.data) return;
    const { frames, fps, keptSteps } = this.meta;
    // The clip's clock stops while the steps change (dragging or "Play steps").
    if (!this.scrubbing && !this.state.playingSteps) this.time += dt;
    if (this.state.playingSteps) {
      this.stepClock += dt;
      if (this.stepClock >= STEP_SECONDS) {
        this.stepClock = 0;
        if (this.state.step < keptSteps.length - 1) this.state.step += 1;
        else this.state.playingSteps = false;
        this.update();
      }
    }
    const s = this.state.step;
    const which = { left: 0, right: 1 };
    const strobe = this.state.view === 'strobe';
    for (const slot of ['left', 'right']) {
      const { main, ghosts } = this.robots[slot];
      if (strobe) {
        const picks = Array.from({ length: STROBE_POSES }, (_, i) => Math.round((i * (frames - 1)) / (STROBE_POSES - 1)));
        ghosts.forEach((g, i) => {
          g.visible = true;
          g.pose(this.data, this.clipOffset(which[slot], s) + picks[i] * this.meta.nq);
        });
        this.pose(slot, which[slot], s, picks[STROBE_POSES - 1]);
      } else {
        ghosts.forEach((g) => { g.visible = false; });
        this.pose(slot, which[slot], s, Math.floor(this.time * fps) % frames);
      }
    }
    this.controls.update();
    this.renderRows();
  }

  /** Render the two rows: the left-slot robot on top, the right-slot robot below. */
  renderRows() {
    const { w, h } = this.size;
    const bar = document.querySelector('.bar').offsetHeight + 24;
    const rowH = Math.max(60, Math.floor((h - bar) / 2));
    this.camera.aspect = w / rowH;
    this.camera.updateProjectionMatrix();
    this.renderer.setScissorTest(true);
    ['left', 'right'].forEach((slot, i) => {
      for (const other of ['left', 'right']) {
        const { main, ghosts } = this.robots[other];
        const show = other === slot;
        main.visible = show;
        if (!show) ghosts.forEach((g) => { g.visible = false; });
      }
      if (this.state.view !== 'strobe') this.robots[slot].ghosts.forEach((g) => { g.visible = false; });
      else this.robots[slot].ghosts.forEach((g) => { g.visible = true; });
      const y = h - (i + 1) * rowH;          // WebGL viewports count from the bottom
      this.renderer.setViewport(0, y, w, rowH);
      this.renderer.setScissor(0, y, w, rowH);
      this.renderer.render(this.scene, this.camera);
    });
    this.renderer.setScissorTest(false);
    document.documentElement.style.setProperty('--row-h', `${rowH}px`);
  }
}

new Demo().init().catch((err) => {
  console.error(err);
  $('error').textContent = `The demo could not start: ${err.message}`;
  $('error').hidden = false;
});
