const PAPER = [0.98, 0.98, 0.976];
const INK = [0.086, 0.086, 0.086];
const ACCENT = [0.114, 0.306, 0.847];

const VS = [
  "attribute vec3 aPos;",
  "uniform mat4 uProj;",
  "uniform vec2 uRot;",
  "uniform vec3 uMouse;",
  "uniform float uTime;",
  "uniform float uBump;",
  "varying float vBump;",
  "varying float vZ;",
  "vec3 rotate(vec3 p, vec2 r) {",
  "  float cy = cos(r.x), sy = sin(r.x);",
  "  float cx = cos(r.y), sx = sin(r.y);",
  "  p = vec3(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);",
  "  p = vec3(p.x, cx * p.y - sx * p.z, sx * p.y + cx * p.z);",
  "  return p;",
  "}",
  "void main() {",
  "  vec3 n = normalize(aPos);",
  "  float d = distance(n, uMouse);",
  "  float bump = exp(-d * d * 2.4) * uBump;",
  "  vBump = bump;",
  "  float wave = sin(uTime * 0.8 + n.y * 2.6 + n.x * 2.0) * 0.03;",
  "  vec3 p = n * (1.0 + wave + bump * 0.3) * 0.86;",
  "  p = rotate(p, uRot);",
  "  vec4 vp = vec4(p + vec3(0.0, 0.0, -3.05), 1.0);",
  "  vZ = vp.z;",
  "  gl_PointSize = 1.5 + bump * 4.5;",
  "  gl_Position = uProj * vp;",
  "}",
].join("\n");

const FS = [
  "precision mediump float;",
  "uniform vec3 uInk;",
  "uniform vec3 uAccent;",
  "uniform vec3 uPaper;",
  "varying float vBump;",
  "varying float vZ;",
  "void main() {",
  "  float depth = smoothstep(-4.1, -2.0, vZ);",
  "  vec3 col = mix(uInk, uAccent, clamp(vBump * 2.4, 0.0, 1.0));",
  "  col = mix(uPaper, col, mix(0.3, 1.0, depth));",
  "  float alpha = mix(0.25, 0.95, depth);",
  "  gl_FragColor = vec4(col, alpha);",
  "}",
].join("\n");

function buildGeodesic() {
  const t = (1 + Math.sqrt(5)) / 2;
  const verts = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ];
  const faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  const cache = new Map();
  function mid(a, b) {
    const key = a < b ? a + "," + b : b + "," + a;
    if (cache.has(key)) return cache.get(key);
    const i = verts.length;
    verts.push([
      (verts[a][0] + verts[b][0]) / 2,
      (verts[a][1] + verts[b][1]) / 2,
      (verts[a][2] + verts[b][2]) / 2,
    ]);
    cache.set(key, i);
    return i;
  }
  const tris = [];
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    const ab = mid(f[0], f[1]);
    const bc = mid(f[1], f[2]);
    const ca = mid(f[2], f[0]);
    tris.push([f[0], ab, ca], [f[1], bc, ab], [f[2], ca, bc], [ab, bc, ca]);
  }
  const positions = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    const x = verts[i][0], y = verts[i][1], z = verts[i][2];
    const l = Math.sqrt(x * x + y * y + z * z) || 1;
    positions[i * 3] = x / l;
    positions[i * 3 + 1] = y / l;
    positions[i * 3 + 2] = z / l;
  }
  const seen = new Set();
  const edges = [];
  for (let i = 0; i < tris.length; i++) {
    for (let j = 0; j < 3; j++) {
      const a = tris[i][j];
      const b = tris[i][(j + 1) % 3];
      const key = a < b ? a + "," + b : b + "," + a;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push(a, b);
      }
    }
  }
  return { positions, edges: new Uint16Array(edges) };
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0,
  ]);
}

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader error");
  return s;
}

export function init(canvas) {
  let gl = null;
  try {
    gl = canvas.getContext("webgl", {
      alpha: true,
      antialias: true,
      premultipliedAlpha: false,
      powerPreference: "low-power",
    });
  } catch (e) {
    gl = null;
  }
  if (!gl) return;

  const geo = buildGeodesic();
  let prog;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link error");
  } catch (e) {
    return;
  }
  gl.useProgram(prog);

  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, geo.positions, gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(prog, "aPos");
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

  const edgeBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, edgeBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geo.edges, gl.STATIC_DRAW);

  const uProj = gl.getUniformLocation(prog, "uProj");
  const uRot = gl.getUniformLocation(prog, "uRot");
  const uMouse = gl.getUniformLocation(prog, "uMouse");
  const uTime = gl.getUniformLocation(prog, "uTime");
  const uBump = gl.getUniformLocation(prog, "uBump");

  gl.uniform3fv(gl.getUniformLocation(prog, "uInk"), INK);
  gl.uniform3fv(gl.getUniformLocation(prog, "uAccent"), ACCENT);
  gl.uniform3fv(gl.getUniformLocation(prog, "uPaper"), PAPER);
  gl.uniform1f(uBump, 0.6);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0, 0, 0, 0);

  const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let tx = 0, ty = 0, mx = 0, my = 0;
  window.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    tx = Math.max(-1.8, Math.min(1.8, ((e.clientX - r.left) / r.width) * 2 - 1));
    ty = Math.max(-1.8, Math.min(1.8, ((e.clientY - r.top) / r.height) * 2 - 1));
  }, { passive: true });
  document.addEventListener("mouseleave", () => { tx = 0; ty = 0; });

  function resize() {
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width === w && canvas.height === h) return;
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
    gl.uniformMatrix4fv(uProj, false, perspective((42 * Math.PI) / 180, w / h, 0.1, 24));
  }
  resize();
  new ResizeObserver(resize).observe(canvas);

  function mouseDir(x, y) {
    const lon = x * 1.15;
    const lat = -y;
    const cl = Math.cos(lat);
    return [Math.sin(lon) * cl, Math.sin(lat), Math.cos(lon) * cl];
  }

  let t = 0;
  let raf = 0;
  let last = performance.now();

  function draw() {
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(uRot, t * 0.1 + mx * 0.3, Math.sin(t * 0.2) * 0.06 + my * 0.3);
    gl.uniform3fv(uMouse, mouseDir(mx, my));
    gl.uniform1f(uTime, t);
    gl.drawElements(gl.LINES, geo.edges.length, gl.UNSIGNED_SHORT, 0);
    gl.drawArrays(gl.POINTS, 0, geo.positions.length / 3);
  }

  if (reduced) {
    t = 1.2;
    draw();
    return;
  }

  function tick(now) {
    raf = requestAnimationFrame(tick);
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    t += dt;
    mx += (tx - mx) * 0.075;
    my += (ty - my) * 0.075;
    draw();
  }
  function start() {
    if (!raf) {
      last = performance.now();
      raf = requestAnimationFrame(tick);
    }
  }
  function stop() {
    if (raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
  }

  let inView = true;
  let visible = !document.hidden;
  function sync() {
    if (inView && visible) start();
    else stop();
  }

  new IntersectionObserver((es) => {
    inView = es[es.length - 1].isIntersecting;
    sync();
  }, { threshold: 0.02 }).observe(canvas);
  document.addEventListener("visibilitychange", () => {
    visible = !document.hidden;
    sync();
  });
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    stop();
  });

  start();
}
