/**
 * car3d.js
 * --------
 * A small procedural, generic low-poly car (no brand, no licensed design —
 * built entirely from primitives) that sits in the hero gauge panel and
 * slowly rotates. Its glow color responds to the fuel type selected in the
 * calculator, so the hero stays connected to the tool below it.
 */
import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

const FUEL_COLORS = {
  Gasoline: 0xffb330,
  Diesel: 0x9aa3b5,
  Hybrid: 0x35d6b3,
  Plug: 0x35d6b3,
  Other: 0xffb330,
};

export function initCar3D(canvas) {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(3.4, 2.1, 4.4);
  camera.lookAt(0, 0.35, 0);

  scene.add(new THREE.AmbientLight(0x88a0c0, 0.7));
  const key = new THREE.DirectionalLight(0xffe4b0, 1.1);
  key.position.set(4, 6, 3);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x35d6b3, 0.6);
  rim.position.set(-4, 2, -3);
  scene.add(rim);

  const carGroup = new THREE.Group();
  scene.add(carGroup);

  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xffb330, metalness: 0.35, roughness: 0.35 });
  const glassMat = new THREE.MeshStandardMaterial({ color: 0x0d1420, metalness: 0.1, roughness: 0.15 });
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x14161b, metalness: 0.2, roughness: 0.8 });
  const trimMat = new THREE.MeshStandardMaterial({ color: 0x20242e, metalness: 0.6, roughness: 0.4 });

  // lower body
  const lowerBody = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.55, 1.5, 2, 1, 1), bodyMat);
  lowerBody.position.y = 0.42;
  carGroup.add(lowerBody);

  // cabin (trapezoid-ish via scaled box)
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.55, 1.36), glassMat);
  cabin.position.set(-0.15, 0.92, 0);
  carGroup.add(cabin);

  // cabin roof trim
  const roof = new THREE.Mesh(new THREE.BoxGeometry(1.55, 0.08, 1.3), trimMat);
  roof.position.set(-0.15, 1.21, 0);
  carGroup.add(roof);

  // bumpers
  const frontBumper = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.32, 1.56), trimMat);
  frontBumper.position.set(1.62, 0.34, 0);
  carGroup.add(frontBumper);
  const rearBumper = frontBumper.clone();
  rearBumper.position.x = -1.62;
  carGroup.add(rearBumper);

  // headlight / taillight accents
  const lightGeo = new THREE.BoxGeometry(0.04, 0.14, 0.3);
  const headMat = new THREE.MeshStandardMaterial({ color: 0xfff3d6, emissive: 0xffdc8a, emissiveIntensity: 0.6 });
  const tailMat = new THREE.MeshStandardMaterial({ color: 0x5a1a1a, emissive: 0xaa2b2b, emissiveIntensity: 0.4 });
  [0.62, -0.62].forEach((z) => {
    const head = new THREE.Mesh(lightGeo, headMat);
    head.position.set(1.68, 0.46, z);
    carGroup.add(head);
    const tail = new THREE.Mesh(lightGeo, tailMat);
    tail.position.set(-1.68, 0.46, z);
    carGroup.add(tail);
  });

  // wheels
  const wheelGeo = new THREE.CylinderGeometry(0.42, 0.42, 0.32, 20);
  const hubMat = new THREE.MeshStandardMaterial({ color: 0x8b93a6, metalness: 0.7, roughness: 0.3 });
  const wheelPositions = [
    [1.05, 0.42, 0.85],
    [1.05, 0.42, -0.85],
    [-1.05, 0.42, 0.85],
    [-1.05, 0.42, -0.85],
  ];
  wheelPositions.forEach(([x, y, z]) => {
    const wheel = new THREE.Mesh(wheelGeo, wheelMat);
    wheel.rotation.x = Math.PI / 2;
    wheel.position.set(x, y, z);
    carGroup.add(wheel);
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.34, 12), hubMat);
    hub.rotation.x = Math.PI / 2;
    hub.position.set(x, y, z);
    carGroup.add(hub);
  });

  // subtle ground shadow disc
  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(2.1, 32),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.28 })
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.01;
  scene.add(shadow);

  carGroup.rotation.y = 0.5;

  function resize() {
    const size = canvas.clientWidth || canvas.parentElement.clientWidth;
    renderer.setSize(size, size, false);
    camera.aspect = 1;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener("resize", resize);

  let raf = null;
  function tick() {
    if (!reduceMotion) carGroup.rotation.y += 0.0035;
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  }
  tick();

  return {
    setFuelColor(fuel) {
      const hex = FUEL_COLORS[fuel] ?? FUEL_COLORS.Gasoline;
      bodyMat.color.setHex(hex);
      rim.color.setHex(hex);
    },
    dispose() {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    },
  };
}
