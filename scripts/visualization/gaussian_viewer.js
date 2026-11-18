// Reused Gaussian renderer; original source and SHA in provenance.json.
const vertexSource="#version 300 es\nprecision highp float;\nprecision highp int;\nlayout(location=0) in uint a_index;\nlayout(location=1) in vec2 a_corner;\nuniform highp sampler2D u_data;\nuniform mat3 u_view;\nuniform vec3 u_center;\nuniform vec2 u_size;\nuniform float u_focal;\nout vec2 v_delta;\nflat out vec4 v_rgba;\n\nmat3 quatRotation(vec4 q) {\n  q = normalize(q);\n  float w=q.x, x=q.y, y=q.z, z=q.w;\n  return mat3(\n    1.0-2.0*(y*y+z*z), 2.0*(x*y+w*z), 2.0*(x*z-w*y),\n    2.0*(x*y-w*z), 1.0-2.0*(x*x+z*z), 2.0*(y*z+w*x),\n    2.0*(x*z+w*y), 2.0*(y*z-w*x), 1.0-2.0*(x*x+y*y)\n  );\n}\n\nvoid main() {\n  int index = int(a_index);\n  ivec2 texel = ivec2((index % 128) * 4, index / 128);\n  vec4 meanOpacity = texelFetch(u_data, texel, 0);\n  vec4 scaleR = texelFetch(u_data, texel + ivec2(1,0), 0);\n  vec4 quat = texelFetch(u_data, texel + ivec2(2,0), 0);\n  vec4 gb = texelFetch(u_data, texel + ivec2(3,0), 0);\n  v_rgba = vec4(scaleR.w, gb.x, gb.y, meanOpacity.w);\n\n  vec3 center = u_view * (meanOpacity.xyz - u_center) + vec3(0.0, 0.0, 1.9);\n  if (center.z <= 0.025 || v_rgba.a < 0.0039215686) {\n    gl_Position = vec4(2.0,2.0,0.0,1.0);\n    v_delta = vec2(0.0);\n    return;\n  }\n\n  vec3 scale = exp2(scaleR.xyz);\n  mat3 rotation = u_view * quatRotation(quat);\n  mat3 axes = mat3(rotation[0]*scale.x, rotation[1]*scale.y, rotation[2]*scale.z);\n  float invZ = 1.0 / center.z;\n  float focalZ = u_focal * invZ;\n  vec3 jacX = vec3(focalZ, 0.0, -focalZ * center.x * invZ);\n  vec3 jacY = vec3(0.0, focalZ, -focalZ * center.y * invZ);\n  vec3 rowX = transpose(axes) * jacX;\n  vec3 rowY = transpose(axes) * jacY;\n  float a = dot(rowX,rowX) + 0.3;\n  float b = dot(rowX,rowY);\n  float d = dot(rowY,rowY) + 0.3;\n  float average = 0.5 * (a+d);\n  float radius = length(vec2(0.5*(a-d), b));\n  vec2 direction = abs(b) > 1e-7\n    ? normalize(vec2(average + radius - d, b))\n    : (a >= d ? vec2(1.0,0.0) : vec2(0.0,1.0));\n  vec2 major = direction * sqrt(max(0.3,average+radius));\n  vec2 minor = vec2(-direction.y,direction.x) * sqrt(max(0.3,average-radius));\n  v_delta = a_corner * 3.33;\n  vec2 screen = focalZ * center.xy + v_delta.x*major + v_delta.y*minor;\n  gl_Position = vec4(screen * 2.0 / u_size, 0.0, 1.0);\n}";
const fragmentSource="#version 300 es\nprecision highp float;\nin vec2 v_delta;\nflat in vec4 v_rgba;\nout vec4 outputColor;\nvoid main() {\n  float r2 = dot(v_delta, v_delta);\n  if (r2 > 11.09) discard;\n  float alpha = min(0.999, v_rgba.a * exp(-0.5 * r2));\n  if (alpha < 0.0039215686) discard;\n  outputColor = vec4(v_rgba.rgb * alpha, alpha);\n}";
function createSplatPanel(canvas) {
  const gl = canvas.getContext("webgl2", {alpha: false, depth: false, antialias: false});
  if (!gl) throw new Error("WebGL 2 is unavailable");

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }
  const vertex = compile(gl.VERTEX_SHADER, vertexSource);
  const fragment = compile(gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  gl.useProgram(program);

  const viewUniform = gl.getUniformLocation(program, "u_view");
  const centerUniform = gl.getUniformLocation(program, "u_center");
  const sizeUniform = gl.getUniformLocation(program, "u_size");
  const focalUniform = gl.getUniformLocation(program, "u_focal");
  gl.uniform1i(gl.getUniformLocation(program, "u_data"), 0);

  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribIPointer(0, 1, gl.UNSIGNED_INT, 4, 0);
  gl.vertexAttribDivisor(0, 1);
  const corners = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, corners);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 8, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0, 0, 0, 1);

  let frame = null;
  let order, depths, lastOrientation;

  function setFrame(next) {
    frame = next;
    const height = Math.ceil(frame.count / 128);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 512, height, 0, gl.RGBA, gl.HALF_FLOAT, frame.data);
    order = Uint32Array.from({length: frame.count}, (_, i) => i);
    depths = new Float32Array(frame.count);
    lastOrientation = null;
  }

  function draw(view, pivot, zoom) {
    if (!frame) return;
    const ratio = Math.min(1.5, window.devicePixelRatio || 1);
    const width = Math.round(canvas.clientWidth * ratio);
    const height = Math.round(canvas.clientHeight * ratio);
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (view !== lastOrientation) {
      const xyz = frame.positions;
      for (let i = 0, j = 0; i < frame.count; i++, j += 3) {
        depths[i] = view[2]*xyz[j] + view[5]*xyz[j+1] + view[8]*xyz[j+2];
      }
      order.sort((a,b) => depths[b]-depths[a]);
      gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
      lastOrientation = view;
    }
    gl.uniformMatrix3fv(viewUniform, false, view);
    gl.uniform3fv(centerUniform, pivot);
    gl.uniform2f(sizeUniform, width, height);
    gl.uniform1f(focalUniform, 0.95 * Math.min(width, height) * zoom);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, frame.count);
  }
  return {setFrame, draw};
}

function orbitMatrix(yaw, pitch) {
  const c=Math.cos(yaw), s=Math.sin(yaw), p=Math.cos(pitch), q=Math.sin(pitch);
  return new Float32Array([c, q*s, -p*s, s, -q*c, p*c, 0, p, q]);
}

function halfToFloat(bits) {
  const exponent = (bits >>> 10) & 31;
  const fraction = bits & 1023;
  const magnitude = exponent === 0 ? fraction * 5.960464477539063e-8 : (1 + fraction / 1024) * 2 ** (exponent - 15);
  return bits & 0x8000 ? -magnitude : magnitude;
}

async function decodeSplat({count, gzip}) {
  const compressed = Uint8Array.from(atob(gzip), ch => ch.charCodeAt(0));
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
  const data = new Uint16Array(await new Response(stream).arrayBuffer());
  if (data.length !== Math.ceil(count / 128) * 128 * 16) throw new Error("Incorrect Gaussian frame size");
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[3*i] = halfToFloat(data[16*i]);
    positions[3*i+1] = halfToFloat(data[16*i+1]);
    positions[3*i+2] = halfToFloat(data[16*i+2]);
  }
  return {count, data, positions};
}

