import React, { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, ThreeEvent, useThree, extend, ThreeElement } from '@react-three/fiber'
import { OrbitControls, Stars, PerspectiveCamera, shaderMaterial, Text, Html, Detailed } from '@react-three/drei'
import { XR, createXRStore, useXR, useXRInputSourceStates } from '@react-three/xr'
import { EffectComposer, Bloom, ChromaticAberration, Vignette, Noise, SSAO } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import * as THREE from 'three'
import { motion, AnimatePresence } from 'motion/react'
import { compile } from 'mathjs'
import 'katex/dist/katex.min.css'
import { BlockMath } from 'react-katex'

// --- 1. True Fluid Shader (Liquid Glass) ---
const FluidGelMaterial = shaderMaterial(
  {
    uTime: 0,
    uPunchPos: new THREE.Vector3(0, 0, 0),
    uPunchTime: -100.0,
    uColorBase: new THREE.Color('#000a1f'),
    uColorSurface: new THREE.Color('#00ffff'),
    uWaveSpeed: 12.0,
    uIntensityFalloff: 0.2,
  },
  // Vertex Shader: Fluid Deformation
  `
    varying vec3 vPos;
    varying vec3 vViewPosition;
    varying vec2 vUv;
    uniform float uTime;
    uniform vec3 uPunchPos;
    uniform float uPunchTime;
    uniform float uWaveSpeed;
    uniform float uIntensityFalloff;

    // Simplex 3D Noise
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
    float snoise(vec3 v) {
      const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
      const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
      vec3 i  = floor(v + dot(v, C.yyy) );
      vec3 x0 = v - i + dot(i, C.xxx) ;
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min( g.xyz, l.zxy );
      vec3 i2 = max( g.xyz, l.zxy );
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy;
      vec3 x3 = x0 - D.yyy;
      i = mod289(i);
      vec4 p = permute( permute( permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
      float n_ = 0.142857142857;
      vec3  ns = n_ * D.wyz - D.xzx;
      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_ );
      vec4 x = x_ *ns.x + ns.yyyy;
      vec4 y = y_ *ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);
      vec4 b0 = vec4( x.xy, y.xy );
      vec4 b1 = vec4( x.zw, y.zw );
      vec4 s0 = floor(b0)*2.0 + 1.0;
      vec4 s1 = floor(b1)*2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));
      vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
      vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
      vec3 p0 = vec3(a0.xy,h.x);
      vec3 p1 = vec3(a0.zw,h.y);
      vec3 p2 = vec3(a1.xy,h.z);
      vec3 p3 = vec3(a1.zw,h.w);
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
      p0 *= norm.x;
      p1 *= norm.y;
      p2 *= norm.z;
      p3 *= norm.w;
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                    dot(p2,x2), dot(p3,x3) ) );
    }

    void main() {
      vUv = uv;
      vec3 pos = position;
      
      // 1. Continuous Fluid Motion (Large, slow rolling waves)
      float noise = snoise(pos * 0.03 + uTime * 0.1) * 4.0;
      pos += normal * noise;

      // 2. The Interactive Punch (Expanding Ripple)
      float dist = distance(pos, uPunchPos);
      float t = uTime - uPunchTime;
      if (t > 0.0 && t < 20.0) {
         float waveFront = t * uWaveSpeed;
         float distToWave = abs(dist - waveFront);
         
         // Gaussian falloff for the ripple
         float intensity = exp(-distToWave * uIntensityFalloff) * exp(-t * 0.15);
         
         // The fluid ripples and twists
         pos += normal * sin(dist * 1.5 - t * 8.0) * 3.0 * intensity;
         
         // Add a slight transverse shear (twist) to the fluid
         pos.x += cos(pos.y * 0.1 + t * 5.0) * 1.5 * intensity;
         pos.y += sin(pos.x * 0.1 + t * 5.0) * 1.5 * intensity;
      }

      vPos = pos;
      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      vViewPosition = -mvPosition.xyz;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  // Fragment Shader: Liquid Glass / Fresnel Effect
  `
    varying vec3 vPos;
    varying vec3 vViewPosition;
    uniform vec3 uColorBase;
    uniform vec3 uColorSurface;

    void main() {
      // Calculate normal dynamically from deformed vertices for a faceted/smooth liquid look
      vec3 dx = dFdx(vPos);
      vec3 dy = dFdy(vPos);
      vec3 normal = normalize(cross(dx, dy));
      
      // Ensure normal faces the camera (since we are inside the sphere)
      vec3 viewDir = normalize(vViewPosition);
      if (dot(normal, viewDir) < 0.0) {
          normal = -normal;
      }

      // Fresnel effect for liquid glass appearance
      float fresnel = 1.0 - max(dot(viewDir, normal), 0.0);
      fresnel = pow(fresnel, 2.5);

      // Mix deep liquid color with bright surface reflections
      vec3 color = mix(uColorBase, uColorSurface, fresnel);
      
      // Specular highlight for wetness
      vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
      vec3 halfVector = normalize(lightDir + viewDir);
      float specular = pow(max(dot(normal, halfVector), 0.0), 80.0);

      // Fade out at the edges to blend into space (silhouette fade)
      float edgeFade = smoothstep(1.0, 0.5, fresnel);

      // Final composite
      vec3 finalColor = color + specular * 1.5;
      float alpha = (0.1 + fresnel * 0.3 + specular) * edgeFade;

      gl_FragColor = vec4(finalColor, alpha);
    }
  `
)

// --- 1.2. Holographic Line Shader ---
const HolographicLineMaterial = shaderMaterial(
  {
    uTime: 0,
    uColor: new THREE.Color('#00ffff'),
    uOpacity: 0.8,
  },
  // Vertex
  `
    varying float vZ;
    void main() {
      vZ = position.z;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  // Fragment
  `
    uniform float uTime;
    uniform vec3 uColor;
    uniform float uOpacity;
    varying float vZ;

    void main() {
      // Holographic scanline effect
      float scanline = sin(vZ * 0.5 - uTime * 10.0) * 0.5 + 0.5;
      // Fade at the far ends (-80 to 80)
      float edgeFade = 1.0 - smoothstep(40.0, 80.0, abs(vZ));
      
      float alpha = (0.2 + scanline * 0.8) * edgeFade * uOpacity;
      gl_FragColor = vec4(uColor, alpha);
    }
  `
)

extend({ FluidGelMaterial, HolographicLineMaterial })

declare global {
  namespace JSX {
    interface IntrinsicElements {
      fluidGelMaterial: ThreeElement<typeof FluidGelMaterial>
      holographicLineMaterial: ThreeElement<typeof HolographicLineMaterial>
    }
  }
}

// --- 1.5. XR Store Setup ---
const store = createXRStore({
  hand: true,
  controller: true,
})

// --- 1.6. Sound Synthesis Utility ---
// Create a persistent audio context and global nodes for ambient sound
let audioCtx: AudioContext | null = null;
let ambientOsc: OscillatorNode | null = null;
let ambientGain: GainNode | null = null;
let modulationOsc: OscillatorNode | null = null;
let modulationGain: GainNode | null = null;

const initAmbientAudio = () => {
  if (audioCtx || !window.AudioContext) return;
  try {
    audioCtx = new AudioContext();
    
    // Carrier oscillator (low hum)
    ambientOsc = audioCtx.createOscillator();
    ambientOsc.type = 'sine';
    ambientOsc.frequency.value = 55; // Low A
    
    // Modulation oscillator for pulsing effect
    modulationOsc = audioCtx.createOscillator();
    modulationOsc.type = 'sine';
    modulationOsc.frequency.value = 0.5; // 0.5 Hz pulse
    
    modulationGain = audioCtx.createGain();
    modulationGain.gain.value = 10; // Modulation depth
    
    ambientGain = audioCtx.createGain();
    ambientGain.gain.value = 0; // Start muted
    
    // Connect AM synthesis
    modulationOsc.connect(modulationGain);
    modulationGain.connect(ambientOsc.frequency); // Modulate frequency slightly
    
    ambientOsc.connect(ambientGain);
    ambientGain.connect(audioCtx.destination);
    
    ambientOsc.start();
    modulationOsc.start();
  } catch (e) {
    console.warn("Ambient audio initialization failed", e);
  }
}

const updateAmbientAudio = (viewMode: string, waveSpeed: number, volume: number) => {
  if (!audioCtx || !ambientOsc || !ambientGain || !modulationOsc) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();
  
  const now = audioCtx.currentTime;
  
  // Base target volume based on settings
  const targetGain = volume * 0.15;
  ambientGain.gain.setTargetAtTime(targetGain, now, 1.0);
  
  // React to viewMode
  if (viewMode === 'fluid') {
    ambientOsc.frequency.setTargetAtTime(55, now, 0.5); // Deep drone
    ambientOsc.type = 'sine';
    modulationOsc.frequency.setTargetAtTime(0.2, now, 0.5);
  } else if (viewMode === 'math') {
    ambientOsc.frequency.setTargetAtTime(110, now, 0.5); // Higher, cleaner tone
    ambientOsc.type = 'triangle';
    modulationOsc.frequency.setTargetAtTime(4.0, now, 0.5); // Faster flutter
  } else if (viewMode === 'hybrid') {
    ambientOsc.frequency.setTargetAtTime(82.41, now, 0.5); // E2
    ambientOsc.type = 'sawtooth';
    modulationOsc.frequency.setTargetAtTime(waveSpeed * 0.2, now, 0.5); // Reacts to speed
  }
}

const playSound = (type: 'punch' | 'beam' | 'ui', volume: number = 0.5) => {
  try {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextClass) return;
    const ctx = new AudioContextClass();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.connect(gain);
    gain.connect(ctx.destination);

    const now = ctx.currentTime;

    if (type === 'punch') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(150, now);
      osc.frequency.exponentialRampToValueAtTime(40, now + 0.5);
      gain.gain.setValueAtTime(volume, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
      osc.start(now);
      osc.stop(now + 0.5);
    } else if (type === 'beam') {
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.linearRampToValueAtTime(880, now + 0.1);
      gain.gain.setValueAtTime(volume * 0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
      osc.start(now);
      osc.stop(now + 0.1);
    } else if (type === 'ui') {
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.exponentialRampToValueAtTime(1200, now + 0.05);
      gain.gain.setValueAtTime(volume * 0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.05);
      osc.start(now);
      osc.stop(now + 0.05);
    }
  } catch (e) {
    console.warn('Audio not supported or blocked', e);
  }
}

// --- 1.7. Haptic Feedback Utility ---
const triggerHaptic = (inputSource: any, type: 'punch' | 'beam' | 'ui', globalIntensity: number) => {
  const actuator = inputSource?.gamepad?.hapticActuators?.[0];
  if (!actuator || globalIntensity <= 0) return;
  
  try {
    if (type === 'punch') {
      actuator.pulse(globalIntensity * 1.0, 120); // Strong, medium duration
    } else if (type === 'beam') {
      actuator.pulse(globalIntensity * 0.3, 40);  // Light, short duration
    } else if (type === 'ui') {
      actuator.pulse(globalIntensity * 0.6, 20);  // Crisp, very short duration
    }
  } catch (e) {
    console.warn('Haptics failed', e);
  }
}

// --- 2. The Continuous Fluid Component (Optimized with LOD) ---
const FluidGelMedium = ({ punchData, settings }: { 
  punchData: { position: THREE.Vector3; time: number } | null;
  settings: any;
}) => {
  // Create a single material instance to share across LOD meshes
  const material = useMemo(() => new FluidGelMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide
  }), [])

  useFrame(({ clock }) => {
    material.uTime = clock.getElapsedTime()
    material.uColorBase.set(settings.colorBase)
    material.uColorSurface.set(settings.colorSurface)
    material.uWaveSpeed = settings.waveSpeed
    material.uIntensityFalloff = settings.intensityFalloff

    if (punchData) {
      material.uPunchPos.copy(punchData.position)
      material.uPunchTime = punchData.time
    }
  })

  return (
    <Detailed distances={[0, 60, 120]}>
      {/* High Detail for close up */}
      <mesh material={material}>
        <icosahedronGeometry args={[80, 64]} />
      </mesh>
      {/* Medium Detail */}
      <mesh material={material}>
        <icosahedronGeometry args={[80, 32]} />
      </mesh>
      {/* Low Detail for distant viewing */}
      <mesh material={material}>
        <icosahedronGeometry args={[80, 16]} />
      </mesh>
    </Detailed>
  )
}

// --- 2.5. Instanced Quantum Particles ---
const QuantumParticles = ({ count = 2000, settings }: { count?: number, settings: any }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const linesRef = useRef<THREE.LineSegments>(null!);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  // Pre-compute random positions and phases
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < count; i++) {
      temp.push({
        x: (Math.random() - 0.5) * 160,
        y: (Math.random() - 0.5) * 160,
        z: (Math.random() - 0.5) * 160,
        speed: Math.random() * 0.5 + 0.1,
        phase: Math.random() * Math.PI * 2
      });
    }
    return temp;
  }, [count]);

  const linePositions = useMemo(() => new Float32Array(count * 2 * 3), [count]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (meshRef.current && linesRef.current) {
      for (let i = 0; i < count; i++) {
        const p = particles[i];
        
        // Current time and past time for the trail
        const T = t * p.speed + p.phase;
        const T_past = (t - 0.5) * p.speed + p.phase; // 0.5s trail length
        
        const cx = p.x + Math.sin(T) * 2;
        const cy = p.y + Math.cos(T) * 2;
        const cz = p.z + Math.sin(t * p.speed * 0.5) * 2;

        const px = p.x + Math.sin(T_past) * 2;
        const py = p.y + Math.cos(T_past) * 2;
        const pz = p.z + Math.sin((t - 0.5) * p.speed * 0.5) * 2;

        // Gentle floating motion
        dummy.position.set(cx, cy, cz);
        // Slowly rotate
        dummy.rotation.set(t * p.speed, t * p.speed, 0);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);

        // Update trail line segment
        linePositions[i * 6 + 0] = cx;
        linePositions[i * 6 + 1] = cy;
        linePositions[i * 6 + 2] = cz;
        linePositions[i * 6 + 3] = px;
        linePositions[i * 6 + 4] = py;
        linePositions[i * 6 + 5] = pz;
      }
      meshRef.current.instanceMatrix.needsUpdate = true;
      linesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  return (
    <group>
      <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
        <tetrahedronGeometry args={[0.5, 0]} />
        <meshBasicMaterial color={settings.colorSurface} transparent opacity={0.4} blending={THREE.AdditiveBlending} depthWrite={false} />
      </instancedMesh>
      <lineSegments ref={linesRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={count * 2} array={linePositions} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color={settings.colorSurface} transparent opacity={0.15} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
    </group>
  );
}

// --- 3. The Mathematical 3D Graph (Geo3D Style) ---
const MathGraph = ({ punchData, settings, mathSettings }: { 
  punchData: { position: THREE.Vector3; time: number } | null;
  settings: any;
  mathSettings: any;
}) => {
  const pointsCount = 1000; // Reduced for VR performance
  const helix1Ref = useRef<THREE.Line>(null!);
  const helix2Ref = useRef<THREE.Line>(null!);
  const groupRef = useRef<THREE.Group>(null!);
  const mat1Ref = useRef<any>(null!);
  const mat2Ref = useRef<any>(null!);

  const positions1 = useMemo(() => new Float32Array(pointsCount * 3), []);
  const positions2 = useMemo(() => new Float32Array(pointsCount * 3), []);

  // Compile math equations
  const compiledX = useMemo(() => { try { return compile(mathSettings.eqX) } catch(e) { return null } }, [mathSettings.eqX])
  const compiledY = useMemo(() => { try { return compile(mathSettings.eqY) } catch(e) { return null } }, [mathSettings.eqY])
  
  // Pre-allocate scope to avoid garbage collection in the render loop
  const scope = useMemo(() => ({ z: 0, t: 0, A: 0, k: mathSettings.k, w: mathSettings.w, phi: mathSettings.phi }), []);

  // Format equation for display with real-time values
  const formatEq = useCallback((eq: string) => {
    return eq.replace(/\bk\b/g, mathSettings.k.toFixed(2))
             .replace(/\bw\b/g, mathSettings.w.toFixed(2))
             .replace(/\bphi\b/g, mathSettings.phi.toFixed(2))
             .replace(/\*/g, '');
  }, [mathSettings.k, mathSettings.w, mathSettings.phi]);

  // Rotate the graph to align with the punch direction
  React.useEffect(() => {
    if (punchData && groupRef.current) {
      const dir = punchData.position.clone().normalize();
      groupRef.current.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir);
    }
  }, [punchData]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    
    if (helix1Ref.current && helix2Ref.current) {
      const pos1 = helix1Ref.current.geometry.attributes.position.array as Float32Array;
      const pos2 = helix2Ref.current.geometry.attributes.position.array as Float32Array;
      
      scope.t = t;
      scope.k = mathSettings.k;
      scope.w = mathSettings.w;
      scope.phi = mathSettings.phi;
      
      for (let i = 0; i < pointsCount; i++) {
        const z = (i / pointsCount) * 160 - 80; // Spread along Z axis from -80 to 80
        
        // Calculate intensity based on the wave disturbance
        let intensity = 0.5; // Increased base intensity so math is always clearly visible
        if (punchData) {
          const dist = Math.abs(80 - z); // Distance from punch (which is at z=80)
          const tPunch = t - punchData.time;
          if (tPunch > 0 && tPunch < 20.0) {
            const waveFront = tPunch * settings.waveSpeed;
            const distToWave = Math.abs(dist - waveFront);
            const pulse = Math.exp(-distToWave * settings.intensityFalloff) * Math.exp(-tPunch * 0.15);
            intensity += pulse * 2.0; // Amplify punch effect
          }
        }

        const amplitude = 12.0 * intensity * (mathSettings.baseA ?? 1.0); // Increased base amplitude with user multiplier
        
        scope.z = z;
        scope.A = amplitude;

        if (compiledX && compiledY) {
          try {
            // Helix 1
            pos1[i * 3] = compiledX.evaluate(scope);
            pos1[i * 3 + 1] = compiledY.evaluate(scope);
            pos1[i * 3 + 2] = z;

            // Helix 2 (offset by PI phase, which is z + PI/k)
            scope.z = z + Math.PI / mathSettings.k;
            pos2[i * 3] = compiledX.evaluate(scope);
            pos2[i * 3 + 1] = compiledY.evaluate(scope);
            pos2[i * 3 + 2] = z;
          } catch (e) {
            // Fallback to zero if equation is invalid
            pos1[i * 3] = 0; pos1[i * 3 + 1] = 0; pos1[i * 3 + 2] = z;
            pos2[i * 3] = 0; pos2[i * 3 + 1] = 0; pos2[i * 3 + 2] = z;
          }
        }
      }
      
      helix1Ref.current.geometry.attributes.position.needsUpdate = true;
      helix2Ref.current.geometry.attributes.position.needsUpdate = true;

      if (mat1Ref.current) {
        mat1Ref.current.uTime = t;
        mat1Ref.current.uColor.set(settings.colorSurface);
      }
      if (mat2Ref.current) {
        mat2Ref.current.uTime = t;
      }
    }
  });

  return (
    <group ref={groupRef}>
      {/* Central Z-Axis (Direction of Propagation) */}
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={2} array={new Float32Array([0,0,-80, 0,0,80])} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#ffffff" opacity={0.2} transparent />
      </line>

      {/* X and Y Axes at origin */}
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={2} array={new Float32Array([-15,0,0, 15,0,0])} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#ff0055" opacity={0.4} transparent />
      </line>
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={2} array={new Float32Array([0,-15,0, 0,15,0])} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#00ff55" opacity={0.4} transparent />
      </line>

      {/* The Helical Waves */}
      {/* @ts-ignore */}
      <line ref={helix1Ref}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={pointsCount} array={positions1} itemSize={3} />
        </bufferGeometry>
        {/* @ts-ignore */}
        <holographicLineMaterial ref={mat1Ref} transparent depthWrite={false} blending={THREE.AdditiveBlending} />
      </line>
      {/* @ts-ignore */}
      <line ref={helix2Ref}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={pointsCount} array={positions2} itemSize={3} />
        </bufferGeometry>
        {/* @ts-ignore */}
        <holographicLineMaterial ref={mat2Ref} uColor={new THREE.Color('#ff00ff')} transparent depthWrite={false} blending={THREE.AdditiveBlending} />
      </line>

      {/* Mathematical Labels */}
      <Text position={[16, 0, 0]} fontSize={1.5} color="#ff0055" anchorX="left">X (Transverse)</Text>
      <Text position={[0, 16, 0]} fontSize={1.5} color="#00ff55" anchorX="center">Y (Transverse)</Text>
      <Text position={[0, 0, 40]} fontSize={1.5} color="#ffffff" anchorX="left">Z (Longitudinal Propagation)</Text>
      
      {/* 3D Holographic Equation Display */}
      <Html position={[0, 20, 0]} center transform sprite distanceFactor={15}>
        <div className="bg-black/80 p-6 rounded-2xl border border-cyan-500/50 backdrop-blur-md text-white pointer-events-none shadow-[0_0_30px_rgba(0,255,255,0.2)] min-w-[320px]">
          <div className="text-cyan-400 text-[10px] tracking-widest uppercase mb-4 text-center border-b border-cyan-500/30 pb-2">Live Symbolic Analysis</div>
          <div className="text-xl flex flex-col gap-2">
            <div>
              <BlockMath math={`x(z,t) = ${mathSettings.eqX.replace(/\*/g, '').replace(/\bw\b/g, '\\omega').replace(/\bphi\b/g, '\\phi')}`} />
              <div className="text-cyan-300/80 scale-75 -mt-2">
                <BlockMath math={`= ${formatEq(mathSettings.eqX)}`} />
              </div>
            </div>
            <div>
              <BlockMath math={`y(z,t) = ${mathSettings.eqY.replace(/\*/g, '').replace(/\bw\b/g, '\\omega').replace(/\bphi\b/g, '\\phi')}`} />
              <div className="text-cyan-300/80 scale-75 -mt-2">
                <BlockMath math={`= ${formatEq(mathSettings.eqY)}`} />
              </div>
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t border-cyan-500/30 grid grid-cols-2 gap-x-4 gap-y-2 text-[10px] font-mono">
            <div className="flex justify-between"><span className="text-cyan-500">k:</span> <span>{mathSettings.k.toFixed(3)}</span></div>
            <div className="flex justify-between"><span className="text-cyan-500">w:</span> <span>{mathSettings.w.toFixed(3)}</span></div>
            <div className="flex justify-between"><span className="text-cyan-500">phi:</span> <span>{mathSettings.phi.toFixed(3)}</span></div>
            <div className="flex justify-between"><span className="text-cyan-500">A:</span> <span>{mathSettings.baseA.toFixed(3)}</span></div>
          </div>
        </div>
      </Html>
    </group>
  )
}

// --- 3.5. Punch Visual Indicator ---
const PunchIndicator = ({ punchData, settings }: { 
  punchData: { position: THREE.Vector3; time: number } | null;
  settings: any;
}) => {
  const groupRef = useRef<THREE.Group>(null!);
  const ringRef = useRef<THREE.Mesh>(null!);
  const coneRef = useRef<THREE.Mesh>(null!);
  const ringMaterialRef = useRef<THREE.MeshBasicMaterial>(null!);
  const coneMaterialRef = useRef<THREE.MeshBasicMaterial>(null!);

  useFrame(({ clock }) => {
    if (!punchData || !groupRef.current) return;
    const t = clock.getElapsedTime() - punchData.time;
    
    if (t > 0 && t < 2.0) {
      // Orient towards center (Z points outward)
      const dir = punchData.position.clone().normalize();
      groupRef.current.position.copy(punchData.position);
      groupRef.current.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir);
      
      // Animate scale and opacity
      const progress = t / 2.0; // 0 to 1
      
      // Ring expands significantly
      const ringScale = 1.0 + progress * 15.0; 
      if (ringRef.current) ringRef.current.scale.set(ringScale, ringScale, ringScale);
      
      // Cone stretches slightly and moves inward
      const coneScale = 1.0 + progress * 2.0;
      if (coneRef.current) {
        coneRef.current.scale.set(1.0 + progress, coneScale, 1.0 + progress);
        coneRef.current.position.set(0, 0, -5 - progress * 10);
      }
      
      const opacity = 1.0 - Math.pow(progress, 0.5); // Fast fade out
      if (ringMaterialRef.current) ringMaterialRef.current.opacity = opacity * 0.8;
      if (coneMaterialRef.current) coneMaterialRef.current.opacity = opacity * 0.4;
    } else {
      if (ringMaterialRef.current) ringMaterialRef.current.opacity = 0;
      if (coneMaterialRef.current) coneMaterialRef.current.opacity = 0;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Expanding Ring */}
      <mesh ref={ringRef}>
        <ringGeometry args={[1.8, 2.0, 64]} />
        <meshBasicMaterial ref={ringMaterialRef} color={settings.colorSurface} transparent opacity={0} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      {/* Directional Cone (points inward to the center) */}
      <mesh ref={coneRef} position={[0, 0, -5]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[2, 10, 32]} />
        <meshBasicMaterial ref={coneMaterialRef} color={settings.colorSurface} transparent opacity={0} depthWrite={false} />
      </mesh>
    </group>
  );
}

// --- 3.8. Waveform Analyzer Overlay ---
const WaveformAnalyzer = ({ mathSettings, punchData }: { mathSettings: any, punchData: any }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Setup a compiled version of the X equation to sample it purely for 2D visualization
  const compiledX = useMemo(() => { try { return compile(mathSettings.eqX) } catch(e) { return null } }, [mathSettings.eqX]);
  const scope = useMemo(() => ({ z: 0, t: 0, A: 0, k: mathSettings.k, w: mathSettings.w, phi: mathSettings.phi }), [mathSettings.k, mathSettings.w, mathSettings.phi]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let t = 0;

    const draw = () => {
      t += 0.016; // Approx delta time for 60fps presentation
      
      // Update scope
      scope.t = t;
      scope.k = mathSettings.k;
      scope.w = mathSettings.w;
      scope.phi = mathSettings.phi;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw Grid
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();

      if (!compiledX) {
        animationId = requestAnimationFrame(draw);
        return;
      }

      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.strokeStyle = '#00ffff';
      ctx.lineWidth = 2;

      // Sample the equation across the canvas width
      for (let i = 0; i < canvas.width; i++) {
        // Map pixel x to virtual 'z' space
        const z = (i / canvas.width) * 20 - 10;
        scope.z = z;
        
        let amplitude = mathSettings.baseA;
        
        // Add punch disturbance to the 2D analyzer too
        if (punchData) {
          const tPunch = t - (punchData.time || 0); // Simplified relative time check
           if (tPunch > 0 && tPunch < 5) {
             amplitude += Math.exp(-tPunch) * 2;
           }
        }
        
        scope.A = amplitude;

        try {
          const val = compiledX.evaluate(scope);
          // Scale value to canvas height
          const y = canvas.height / 2 - (val * (canvas.height / 4));
          ctx.lineTo(i, y);
        } catch (e) {
          // Skip drawing this point if evaluation fails
        }
      }
      ctx.stroke();
      
      // Draw glow
      ctx.shadowBlur = 10;
      ctx.shadowColor = '#00ffff';
      ctx.stroke();
      ctx.shadowBlur = 0;

      animationId = requestAnimationFrame(draw);
    };

    draw();

    return () => cancelAnimationFrame(animationId);
  }, [mathSettings, compiledX, punchData, scope]);

  return (
    <div className="absolute bottom-4 right-[350px] w-64 h-24 bg-black/50 backdrop-blur-md border border-cyan-500/30 rounded-lg p-2 overflow-hidden z-[90] pointer-events-none hidden md:block group">
      <div className="text-[8px] text-cyan-500 uppercase tracking-widest mb-1 opacity-50">Local Waveform Sample</div>
      <canvas ref={canvasRef} width={240} height={70} className="w-full h-[70px]" />
      
      {/* Tooltip for Analyzer */}
      <div className="absolute inset-0 bg-black/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
         <span className="text-[10px] text-cyan-300 px-4 text-center">Real-time projection of X-axis equation x(z,t)</span>
      </div>
    </div>
  );
}

// --- 4. The Scene Controller ---
const VRControlPanel = ({ settings, setSettings, setViewMode }: { settings: any, setSettings: any, setViewMode: any }) => {
  const session = useXR((state) => state.session)
  if (!session) return null;

  return (
    <group position={[0, 1.5, -2]}>
      {/* Background Panel */}
      <mesh position={[0, 0, -0.05]}>
        <boxGeometry args={[2.5, 1.5, 0.05]} />
        <meshBasicMaterial color="#001122" transparent opacity={0.8} />
      </mesh>
      
      <Text position={[0, 0.6, 0]} fontSize={0.12} color="#00ffff" anchorX="center">VR Control Station</Text>
      
      {/* Wave Speed Control */}
      <Text position={[-0.8, 0.3, 0]} fontSize={0.08} color="white" anchorX="left">Wave Speed: {settings.waveSpeed.toFixed(1)}</Text>
      <mesh position={[0.4, 0.3, 0]} onClick={() => setSettings({...settings, waveSpeed: Math.max(1, settings.waveSpeed - 2)})}>
         <boxGeometry args={[0.2, 0.2, 0.05]} /><meshBasicMaterial color="#ff0055" />
      </mesh>
      <Text position={[0.4, 0.3, 0.03]} fontSize={0.1} color="white" anchorX="center" pointerEvents="none">-</Text>
      
      <mesh position={[0.8, 0.3, 0]} onClick={() => setSettings({...settings, waveSpeed: Math.min(30, settings.waveSpeed + 2)})}>
         <boxGeometry args={[0.2, 0.2, 0.05]} /><meshBasicMaterial color="#00ff55" />
      </mesh>
      <Text position={[0.8, 0.3, 0.03]} fontSize={0.1} color="white" anchorX="center" pointerEvents="none">+</Text>

      {/* Intensity Control */}
      <Text position={[-0.8, 0.0, 0]} fontSize={0.08} color="white" anchorX="left">Intensity: {settings.intensityFalloff.toFixed(2)}</Text>
      <mesh position={[0.4, 0.0, 0]} onClick={() => setSettings({...settings, intensityFalloff: Math.max(0.01, settings.intensityFalloff - 0.05)})}>
         <boxGeometry args={[0.2, 0.2, 0.05]} /><meshBasicMaterial color="#ff0055" />
      </mesh>
      <Text position={[0.4, 0.0, 0.03]} fontSize={0.1} color="white" anchorX="center" pointerEvents="none">-</Text>
      
      <mesh position={[0.8, 0.0, 0]} onClick={() => setSettings({...settings, intensityFalloff: Math.min(1.0, settings.intensityFalloff + 0.05)})}>
         <boxGeometry args={[0.2, 0.2, 0.05]} /><meshBasicMaterial color="#00ff55" />
      </mesh>
      <Text position={[0.8, 0.0, 0.03]} fontSize={0.1} color="white" anchorX="center" pointerEvents="none">+</Text>

      {/* Auto Evolve Toggle */}
      <Text position={[-0.8, -0.3, 0]} fontSize={0.08} color="white" anchorX="left">Auto Evolve Wave</Text>
      <mesh position={[0.6, -0.3, 0]} onClick={() => setSettings({...settings, autoEvolve: !settings.autoEvolve})}>
         <boxGeometry args={[0.6, 0.2, 0.05]} /><meshBasicMaterial color={settings.autoEvolve ? "#00ff55" : "#333333"} />
      </mesh>
      <Text position={[0.6, -0.3, 0.03]} fontSize={0.08} color="white" anchorX="center" pointerEvents="none">{settings.autoEvolve ? 'ON' : 'OFF'}</Text>
      
      {/* Visual Modes */}
      <Text position={[-0.8, -0.6, 0]} fontSize={0.08} color="white" anchorX="left">View Mode</Text>
      <mesh position={[0.0, -0.6, 0]} onClick={() => setViewMode('fluid')}>
         <boxGeometry args={[0.3, 0.15, 0.05]} /><meshBasicMaterial color="#00ffff" />
      </mesh>
      <Text position={[0.0, -0.6, 0.03]} fontSize={0.06} color="black" anchorX="center" pointerEvents="none">F</Text>
      <mesh position={[0.4, -0.6, 0]} onClick={() => setViewMode('math')}>
         <boxGeometry args={[0.3, 0.15, 0.05]} /><meshBasicMaterial color="#ff00ff" />
      </mesh>
      <Text position={[0.4, -0.6, 0.03]} fontSize={0.06} color="black" anchorX="center" pointerEvents="none">M</Text>
      <mesh position={[0.8, -0.6, 0]} onClick={() => setViewMode('hybrid')}>
         <boxGeometry args={[0.3, 0.15, 0.05]} /><meshBasicMaterial color="#ffff00" />
      </mesh>
      <Text position={[0.8, -0.6, 0.03]} fontSize={0.06} color="black" anchorX="center" pointerEvents="none">H</Text>
    </group>
  );
}

const Volumetric3DWaveform = ({ settings, mathSettings }: { settings: any, mathSettings: any }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const count = 30; // 30x30 grid = 900 instances
  
  const compiledX = useMemo(() => { try { return compile(mathSettings.eqX) } catch { return null } }, [mathSettings.eqX]);
  const scope = useMemo(() => ({ z:0, t:0, A:0, k: mathSettings.k, w: mathSettings.w, phi: mathSettings.phi }), [mathSettings.k, mathSettings.w, mathSettings.phi]);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useFrame(({ clock }) => {
    if (!settings.show3DWaveform || !meshRef.current || !compiledX) return;
    
    const t = clock.getElapsedTime();
    scope.t = t;
    scope.k = settings.autoEvolve ? mathSettings.k + Math.sin(t * 0.2) * 0.1 : mathSettings.k;
    scope.w = mathSettings.w;
    scope.phi = settings.autoEvolve ? mathSettings.phi + t * 0.5 : mathSettings.phi;

    let idx = 0;
    for (let x = 0; x < count; x++) {
      for (let z = 0; z < count; z++) {
        // Map 0-30 to physical space (-20 to 20)
        const px = (x / count) * 40 - 20;
        const pz = (z / count) * 40 - 20;
        
        scope.z = pz;
        // Dampen amplitude at edges for a cleaner look
        scope.A = mathSettings.baseA * Math.exp(-Math.pow(px/10, 2));
        
        let py = 0;
        try {
          py = compiledX.evaluate(scope);
        } catch { py = 0; }
        
        dummy.position.set(px, py - 40, pz); // Offset down so it doesn't block main fluid
        dummy.scale.set(0.5, Math.max(0.1, Math.abs(py) * 2), 0.5);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(idx++, dummy.matrix);
      }
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  if (!settings.show3DWaveform) return null;

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count * count]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={new THREE.Color(settings.colorSurface)} transparent opacity={0.6} roughness={0.2} metalness={0.8} />
    </instancedMesh>
  );
}

const Scene = ({ viewMode, setViewMode, settings, setSettings, mathSettings, setPunchDataState }: { 
  viewMode: string; 
  setViewMode: (mode: string) => void;
  settings: any;
  setSettings: any;
  mathSettings: any;
  setPunchDataState?: (data: any) => void;
}) => {
  const [punchData, setPunchData] = useState<{ position: THREE.Vector3; time: number } | null>(null)
  const [activeBeams, setActiveBeams] = useState<{ id: string; position: THREE.Vector3 }[]>([])
  
  useEffect(() => {
    if (setPunchDataState) {
        setPunchDataState(punchData);
    }
  }, [punchData, setPunchDataState]);
  const { clock } = useThree()

  const handlePunch = useCallback((position: THREE.Vector3, inputSource?: any) => {
    setPunchData({
      position: position,
      time: clock.getElapsedTime()
    })

    // Sound effect
    playSound('punch', settings.soundVolume)

    // Nuanced Haptic feedback
    triggerHaptic(inputSource, 'punch', settings.hapticIntensity)
  }, [clock, settings.hapticIntensity, settings.soundVolume])

  const onPointerDown = useCallback((event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    handlePunch(event.point)
  }, [handlePunch])

  // VR Controller/Hand Interaction
  const inputSources = useXRInputSourceStates()

  // Debounce button presses for mode cycling
  const lastButtonPress = useRef(0)
  const beamActive = useRef<boolean[]>([])

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const beams: { id: string; position: THREE.Vector3 }[] = []
    
    inputSources.forEach((inputSource, index) => {
      // @ts-ignore
      const gamepad = inputSource.gamepad
      // @ts-ignore
      const object = inputSource.object

      // Trigger or Squeeze for vortex punch
      const isPressed = gamepad?.['trigger-button']?.pressed || gamepad?.['squeeze-button']?.pressed;
      
      if (isPressed) {
        const pos = new THREE.Vector3()
        object.getWorldPosition(pos)
        
        // Sound and haptics for beam activation (only on start)
        if (!beamActive.current[index]) {
          playSound('beam', settings.soundVolume)
          triggerHaptic(inputSource, 'beam', settings.hapticIntensity)
          beamActive.current[index] = true
        }

        // Add to active beams for visual feedback
        beams.push({ id: `beam-${index}`, position: pos.clone() })

        // Project onto the fluid sphere (radius 80)
        const punchPos = pos.clone().normalize().multiplyScalar(80)
        handlePunch(punchPos, inputSource)
      } else {
        beamActive.current[index] = false
      }

      // Cycle view modes with A or X button (debounced)
      if (t - lastButtonPress.current > 0.5) {
        if (gamepad?.['a-button']?.pressed || gamepad?.['x-button']?.pressed) {
          const modes = ['fluid', 'math', 'hybrid']
          const currentIndex = modes.indexOf(viewMode)
          const nextIndex = (currentIndex + 1) % modes.length
          setViewMode(modes[nextIndex])
          playSound('ui', settings.soundVolume)
          triggerHaptic(inputSource, 'ui', settings.hapticIntensity)
          lastButtonPress.current = t
        }
      }
    })

    setActiveBeams(beams)
  })

  return (
    <>
      <OrbitControls makeDefault minDistance={5} maxDistance={60} />
      
      {/* Background Space */}
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

      <group onPointerDown={onPointerDown}>
        {/* Render layers based on dropdown selection */}
        {(viewMode === 'fluid' || viewMode === 'hybrid') && (
          <>
            <FluidGelMedium punchData={punchData} settings={settings} />
            <QuantumParticles count={1500} settings={settings} />
          </>
        )}
        {(viewMode === 'math' || viewMode === 'hybrid') && <MathGraph punchData={punchData} settings={settings} mathSettings={mathSettings} />}
        
        <Volumetric3DWaveform settings={settings} mathSettings={mathSettings} />
        <VRControlPanel settings={settings} setSettings={setSettings} setViewMode={setViewMode} />
        
        {/* Invisible sphere to catch clicks across the volume */}
        <mesh>
          <sphereGeometry args={[80, 32, 32]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      </group>

      <PunchIndicator punchData={punchData} settings={settings} />

      {/* Vortex Beams Visual Feedback */}
      {activeBeams.map((beam) => (
        <group key={beam.id}>
          <line>
            <bufferGeometry>
              <bufferAttribute 
                attach="attributes-position" 
                count={2} 
                array={new Float32Array([
                  beam.position.x, beam.position.y, beam.position.z,
                  beam.position.x * 2, beam.position.y * 2, beam.position.z * 2 // Extend outward
                ])} 
                itemSize={3} 
              />
            </bufferGeometry>
            <lineBasicMaterial color={settings.colorSurface} transparent opacity={0.8} linewidth={3} />
          </line>
          <pointLight position={beam.position} color={settings.colorSurface} intensity={5} distance={20} />
        </group>
      ))}

      {/* Visual feedback for controllers/hands (Glow at tips) */}
      {inputSources.map((inputSource, i) => (
        // @ts-ignore
        <primitive key={i} object={inputSource.object}>
          <mesh position={[0, 0, 0]}>
            <sphereGeometry args={[0.03, 16, 16]} />
            <meshBasicMaterial color={settings.colorSurface} transparent opacity={0.8} />
          </mesh>
          <pointLight color={settings.colorSurface} intensity={1} distance={5} />
        </primitive>
      ))}

      {/* Post-processing for Dazzling Visuals - Optimized for VR */}
      <EffectComposer>
        {settings.enableAdvancedRendering && (
          <>
            <SSAO samples={21} radius={0.1} intensity={20} luminanceInfluence={0.1} />
          </>
        )}
        <Bloom 
          luminanceThreshold={0.5} 
          luminanceSmoothing={0.9} 
          intensity={1.0} 
          mipmapBlur 
        />
        {viewMode !== 'fluid' && (
          <ChromaticAberration
            blendFunction={BlendFunction.NORMAL} 
            offset={new THREE.Vector2(0.002, 0.002)} 
          />
        )}
        <Vignette eskil={false} offset={0.1} darkness={1.1} />
        <Noise opacity={0.03} />
      </EffectComposer>
    </>
  )
}

import { Settings, Sliders, Volume2, Zap, Palette, X, Calculator, HelpCircle, ChevronRight, ChevronLeft, Save } from 'lucide-react'

// --- 4.5. Tutorial Component ---
const TutorialOverlay = ({ isOpen, onClose, viewMode, setViewMode, setIsSettingsOpen }: { 
  isOpen: boolean; 
  onClose: () => void;
  viewMode: string;
  setViewMode: (mode: string) => void;
  setIsSettingsOpen: (open: boolean) => void;
}) => {
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: "Welcome to Cosmic Gel Lab",
      content: "Experience the universe as a viscoelastic medium. This lab allows you to interact with longitudinal waves and transverse shear vortices.",
      target: "center"
    },
    {
      title: "Interact with the Fluid",
      content: "Click or tap anywhere on the central sphere to create a vortex punch. In VR, use your trigger or squeeze buttons.",
      target: "center"
    },
    {
      title: "Switching View Layers",
      content: "Hover over the top of the screen to switch between Cosmic Fluid, Mathematical Model, and Hybrid views.",
      target: "top",
      action: () => setViewMode('hybrid')
    },
    {
      title: "Symbolic Math Lab",
      content: "In Math or Hybrid modes, use the left panel to define custom equations. Watch the 3D geometry update in real-time!",
      target: "left",
      action: () => setViewMode('math')
    },
    {
      title: "Fine-Tune the Physics",
      content: "Open the Settings panel in the top right to adjust wave speed, colors, and immersion settings like haptics.",
      target: "right",
      action: () => { setViewMode('fluid'); setIsSettingsOpen(true); }
    },
    {
      title: "Ready for Launch",
      content: "You're all set! Use the 'Enter VR' button for a fully immersive experience on supported devices.",
      target: "center",
      action: () => setIsSettingsOpen(false)
    }
  ];

  if (!isOpen) return null;

  const currentStep = steps[step];

  return (
    <div className="fixed inset-0 z-[1000] pointer-events-none">
      <AnimatePresence>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/40 backdrop-blur-[2px] pointer-events-auto"
          onClick={onClose}
        />
      </AnimatePresence>

      <motion.div
        key={step}
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ 
          opacity: 1, 
          scale: 1, 
          y: 0,
          x: currentStep.target === 'left' ? 'calc(50% - 300px)' : currentStep.target === 'right' ? 'calc(50% + 300px)' : '0%',
          top: currentStep.target === 'top' ? '160px' : '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)'
        }}
        className="absolute w-[400px] bg-black/90 border border-cyan-500/50 p-8 rounded-3xl shadow-[0_0_50px_rgba(0,255,255,0.2)] pointer-events-auto backdrop-blur-xl"
      >
        <div className="flex justify-between items-start mb-4">
          <div className="text-cyan-400 text-[10px] tracking-[0.2em] uppercase font-bold">Tutorial // Step {step + 1} of {steps.length}</div>
          <button onClick={onClose} className="text-white/30 hover:text-white transition-colors"><X size={16} /></button>
        </div>
        
        <h2 className="text-2xl font-light mb-4 text-white tracking-tight">{currentStep.title}</h2>
        <p className="text-sm text-cyan-100/70 leading-relaxed mb-8">{currentStep.content}</p>

        <div className="flex justify-between items-center">
          <button 
            onClick={onClose}
            className="text-[10px] uppercase tracking-widest text-white/30 hover:text-white transition-colors"
          >
            Skip Tutorial
          </button>
          
          <div className="flex gap-2">
            {step > 0 && (
              <button 
                onClick={() => setStep(s => s - 1)}
                className="p-2 bg-white/5 hover:bg-white/10 rounded-full text-white transition-all"
              >
                <ChevronLeft size={20} />
              </button>
            )}
            <button 
              onClick={() => {
                if (step < steps.length - 1) {
                  if (steps[step + 1].action) steps[step + 1].action!();
                  setStep(s => s + 1);
                } else {
                  onClose();
                }
              }}
              className="flex items-center gap-2 px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-full text-xs font-bold tracking-widest uppercase transition-all shadow-[0_0_20px_rgba(8,145,178,0.4)]"
            >
              {step === steps.length - 1 ? "Get Started" : "Next"}
              <ChevronRight size={16} />
            </button>
          </div>
        </div>

        {/* Target Indicator */}
        {currentStep.target !== 'center' && (
          <motion.div 
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.5, 1, 0.5]
            }}
            transition={{ repeat: Infinity, duration: 2 }}
            className={`absolute w-4 h-4 bg-cyan-500 rounded-full shadow-[0_0_20px_cyan] ${
              currentStep.target === 'top' ? '-top-12 left-1/2 -translate-x-1/2' :
              currentStep.target === 'left' ? 'top-1/2 -left-12 -translate-y-1/2' :
              'top-1/2 -right-12 -translate-y-1/2'
            }`}
          />
        )}
      </motion.div>
    </div>
  );
};

// --- 4.6. WebGL Fallback Component ---
const WebGLFallback = () => (
  <div className="flex flex-col items-center justify-center h-full w-full bg-black p-10 text-center">
    <div className="w-20 h-20 mb-6 rounded-full bg-red-500/20 flex items-center justify-center border border-red-500/50">
      <Zap size={40} className="text-red-500" />
    </div>
    <h2 className="text-3xl font-light text-white mb-4 tracking-tight">WebGL Context Error</h2>
    <p className="text-cyan-100/60 max-w-md leading-relaxed mb-8">
      We couldn't initialize the 3D engine. This usually happens if hardware acceleration is disabled in your browser settings or if your graphics drivers are out of date.
    </p>
    <div className="flex flex-col gap-4 text-sm text-left bg-white/5 p-6 rounded-2xl border border-white/10">
      <h4 className="text-cyan-400 uppercase tracking-widest text-[10px] font-bold mb-2">How to fix:</h4>
      <ul className="list-disc list-inside space-y-2 text-white/70">
        <li>Enable <span className="text-white">Hardware Acceleration</span> in browser settings.</li>
        <li>Check if <span className="text-white">WebGL</span> is enabled at <a href="https://get.webgl.org" target="_blank" className="underline text-cyan-400">get.webgl.org</a>.</li>
        <li>Update your graphics card drivers.</li>
        <li>Try a different browser (Chrome or Edge recommended).</li>
      </ul>
    </div>
    <button 
      onClick={() => window.location.reload()} 
      className="mt-10 px-8 py-3 bg-white/10 hover:bg-white/20 text-white rounded-full transition-all border border-white/10 uppercase tracking-widest text-xs font-bold"
    >
      Retry Connection
    </button>
  </div>
);

// Helper component specifically for running side-effects based on deps
const Effect = ({ updater, deps }: { updater: () => void, deps: any[] }) => {
  useEffect(() => {
    updater()
  }, deps)
  return null
}

// --- 5. The Main App with UI ---
export default function App() {
  const [isHovered, setIsHovered] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isTutorialOpen, setIsTutorialOpen] = useState(false)
  const [viewMode, setViewMode] = useState('fluid') // 'fluid', 'math', 'hybrid'
  const [xrSupported, setXrSupported] = useState<boolean | null>(null)
  const [xrError, setXrError] = useState<string | null>(null)
  const [webGLSupported, setWebGLSupported] = useState<boolean | null>(null)
  const [punchDataState, setPunchDataState] = useState<any>(null)

  useEffect(() => {
    // WebGL Support Check
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      setWebGLSupported(!!gl);
    } catch (e) {
      setWebGLSupported(false);
    }

    // XR Support Check
    if ('xr' in navigator) {
      (navigator as any).xr.isSessionSupported('immersive-vr').then((supported: boolean) => {
        setXrSupported(supported)
      }).catch(() => setXrSupported(false))
    } else {
      setXrSupported(false)
    }

    // Tutorial Initialization
    const hasSeenTutorial = localStorage.getItem('cosmic-gel-tutorial-seen');
    if (!hasSeenTutorial) {
      setIsTutorialOpen(true);
      localStorage.setItem('cosmic-gel-tutorial-seen', 'true');
    }
  }, []);

  const [settings, setSettings] = useState({
    colorBase: '#000a1f',
    colorSurface: '#00ffff',
    waveSpeed: 12.0,
    intensityFalloff: 0.2,
    soundVolume: 0.5,
    hapticIntensity: 0.6,
    autoEvolve: false,
    show3DWaveform: false,
    enableAdvancedRendering: false
  })

  // State Management presets
  const [presets, setPresets] = useState<any[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('cosmic-gel-presets') || '[]')
    } catch(e) {
      return []
    }
  })

  const savePreset = () => {
    const newPreset = { id: Date.now(), name: `State ${presets.length + 1}`, settings, mathSettings }
    const updated = [...presets, newPreset]
    setPresets(updated)
    localStorage.setItem('cosmic-gel-presets', JSON.stringify(updated))
  }

  const loadPreset = (p: any) => {
    setSettings(p.settings)
    setMathSettings(p.mathSettings)
  }

  const [mathSettings, setMathSettings] = useState({
    eqX: 'cos(k * z - w * t + phi) * A',
    eqY: 'sin(k * z - w * t + phi) * A',
    k: 0.3,
    w: 3.0,
    baseA: 1.0,
    phi: 0.0
  })

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))
    if (key !== 'soundVolume' && key !== 'hapticIntensity') {
      // playSound('ui', settings.soundVolume)
    }
  }

  return (
    <div className="w-screen h-screen bg-[#000000] overflow-hidden relative font-sans text-white">
      <div className="absolute top-4 right-4 z-[100] flex gap-4">
        <button 
          onClick={() => {
            setIsTutorialOpen(true)
            playSound('ui', settings.soundVolume)
          }}
          className="p-3 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-all border border-white/10"
          title="Show Tutorial"
        >
          <HelpCircle size={20} />
        </button>
        <button 
          onClick={() => {
            setIsSettingsOpen(!isSettingsOpen)
            playSound('ui', settings.soundVolume)
          }}
          className="p-3 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-all border border-white/10 group relative"
          title="Open Settings"
        >
          <Settings size={20} />
          <span className="absolute top-full mt-2 right-0 bg-black/90 text-[10px] tracking-widest uppercase px-2 py-1 border border-white/10 rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity pointer-events-none">
            Lab Settings
          </span>
        </button>
        <button 
          onClick={async () => {
            try {
              setXrError(null)
              initAmbientAudio() // Ensure audio context is started on user gesture
              await store.enterVR()
              playSound('ui', settings.soundVolume)
            } catch (e: any) {
              console.error("Failed to enter VR:", e)
              setXrError(e.message || "VR session failed")
              // Clear error after 5 seconds
              setTimeout(() => setXrError(null), 5000)
            }
          }}
          disabled={xrSupported === false}
          className={`px-6 py-3 rounded-lg font-bold tracking-widest uppercase shadow-lg transition-all relative ${
            xrSupported === false 
              ? 'bg-gray-600 cursor-not-allowed opacity-50' 
              : 'bg-cyan-600 hover:bg-cyan-500 text-white'
          }`}
        >
          {xrSupported === false ? 'VR Not Supported' : 'Enter VR'}
          
          {xrError && (
            <div className="absolute top-full mt-2 right-0 bg-red-500/90 text-white text-[10px] p-2 rounded shadow-xl whitespace-nowrap z-[200]">
              {xrError}
            </div>
          )}
        </button>
      </div>

      <TutorialOverlay 
        isOpen={isTutorialOpen} 
        onClose={() => setIsTutorialOpen(false)} 
        viewMode={viewMode}
        setViewMode={setViewMode}
        setIsSettingsOpen={setIsSettingsOpen}
      />

      {/* Ambient Audio Updater */}
      <Effect updater={() => {
         // This runs when viewMode or settings change, but we need to ensure audioCtx is initialized
         // Usually done on first click in the app to comply with browser autoplay policies
         updateAmbientAudio(viewMode, settings.waveSpeed, settings.soundVolume)
      }} deps={[viewMode, settings.waveSpeed, settings.soundVolume]} />

      {webGLSupported === false ? (
        <WebGLFallback />
      ) : (
        <>
          <Canvas camera={{ position: [0, 10, 30], fov: 45 }} dpr={[1, 1.5]}>
            <XR store={store}>
              <color attach="background" args={['#000000']} />
              <Scene viewMode={viewMode} setViewMode={setViewMode} settings={settings} setSettings={setSettings} mathSettings={mathSettings} setPunchDataState={setPunchDataState} />
            </XR>
          </Canvas>
          {/* Invisible overlay for audio init on first click */}
          {!audioCtx && (
            <div className="absolute inset-0 z-0 pointer-events-auto" onPointerDown={initAmbientAudio} />
          )}
        </>
      )}

      {/* Math Editor Panel */}
      <AnimatePresence>
        {(viewMode === 'math' || viewMode === 'hybrid') && (
          <motion.div 
            initial={{ x: '-100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '-100%', opacity: 0 }}
            className="absolute top-32 left-4 w-80 bg-black/80 backdrop-blur-xl z-[100] border border-cyan-500/30 p-6 rounded-2xl flex flex-col gap-4 shadow-[20px_0_50px_rgba(0,0,0,0.5)]"
          >
            <h3 className="text-cyan-400 tracking-widest uppercase font-bold text-sm flex items-center gap-2">
              <Calculator size={18} /> Symbolic Math Lab
            </h3>
            
            <div className="flex flex-col gap-2">
              <label className="text-[10px] uppercase tracking-widest text-white/50">X-Axis Equation x(z,t)</label>
              <input 
                type="text" 
                value={mathSettings.eqX}
                onChange={(e) => setMathSettings({...mathSettings, eqX: e.target.value})}
                className="bg-black/50 border border-cyan-500/30 rounded p-2 text-sm font-mono text-cyan-300 focus:outline-none focus:border-cyan-400"
              />
            </div>

            <div className="flex flex-col gap-2">
              <label className="text-[10px] uppercase tracking-widest text-white/50">Y-Axis Equation y(z,t)</label>
              <input 
                type="text" 
                value={mathSettings.eqY}
                onChange={(e) => setMathSettings({...mathSettings, eqY: e.target.value})}
                className="bg-black/50 border border-cyan-500/30 rounded p-2 text-sm font-mono text-cyan-300 focus:outline-none focus:border-cyan-400"
              />
            </div>

            <div className="flex flex-col gap-2 mt-2">
              <label className="text-[10px] uppercase tracking-widest text-white/50">Presets</label>
              <div className="grid grid-cols-2 gap-2">
                <button onClick={() => { setMathSettings({...mathSettings, eqX: 'cos(k*z - w*t + phi)*A', eqY: 'sin(k*z - w*t + phi)*A'}); playSound('ui', settings.soundVolume); }} className="text-[10px] p-2 border border-white/10 rounded hover:bg-white/10 transition-colors">Helical</button>
                <button onClick={() => { setMathSettings({...mathSettings, eqX: 'cos(k*z + phi)*cos(w*t)*A', eqY: 'sin(k*z + phi)*cos(w*t)*A'}); playSound('ui', settings.soundVolume); }} className="text-[10px] p-2 border border-white/10 rounded hover:bg-white/10 transition-colors">Standing</button>
                <button onClick={() => { setMathSettings({...mathSettings, eqX: 'sin(k*z - w*t + phi)*A', eqY: '0'}); playSound('ui', settings.soundVolume); }} className="text-[10px] p-2 border border-white/10 rounded hover:bg-white/10 transition-colors">Transverse 2D</button>
                <button onClick={() => { setMathSettings({...mathSettings, eqX: 'cos(k*z - w*t + phi)*A + sin(z)*2', eqY: 'sin(k*z - w*t + phi)*A + cos(z)*2'}); playSound('ui', settings.soundVolume); }} className="text-[10px] p-2 border border-white/10 rounded hover:bg-white/10 transition-colors">Interference</button>
              </div>
            </div>

            <div className="flex flex-col gap-4 mt-2 border-t border-cyan-500/30 pt-4">
              <div className="flex justify-between items-center">
                <label className="text-[10px] uppercase tracking-widest text-white/50">Interactive Parameters</label>
                <button 
                  onClick={() => setMathSettings({
                    eqX: 'cos(k * z - w * t + phi) * A',
                    eqY: 'sin(k * z - w * t + phi) * A',
                    k: 0.3,
                    w: 3.0,
                    baseA: 1.0,
                    phi: 0.0
                  })}
                  className="text-[8px] uppercase tracking-tighter text-cyan-500 hover:text-cyan-400 border border-cyan-500/30 px-2 py-0.5 rounded"
                >
                  Reset
                </button>
              </div>
              
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] text-cyan-300">
                  <span>Wavenumber (k)</span>
                  <span className="font-mono">{mathSettings.k.toFixed(2)}</span>
                </div>
                <div className="relative h-1 bg-white/10 rounded-full overflow-hidden mb-1">
                  <motion.div 
                    className="absolute top-0 left-0 h-full bg-cyan-500"
                    animate={{ width: `${((mathSettings.k - 0.05) / (1.0 - 0.05)) * 100}%` }}
                  />
                </div>
                <input type="range" min="0.05" max="1.0" step="0.01" value={mathSettings.k} onChange={(e) => setMathSettings({...mathSettings, k: parseFloat(e.target.value)})} className="accent-cyan-500 w-full" />
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] text-cyan-300">
                  <span>Angular Freq (w)</span>
                  <span className="font-mono">{mathSettings.w.toFixed(2)}</span>
                </div>
                <div className="relative h-1 bg-white/10 rounded-full overflow-hidden mb-1">
                  <motion.div 
                    className="absolute top-0 left-0 h-full bg-cyan-500"
                    animate={{ width: `${(mathSettings.w / 10.0) * 100}%` }}
                  />
                </div>
                <input type="range" min="0.0" max="10.0" step="0.1" value={mathSettings.w} onChange={(e) => setMathSettings({...mathSettings, w: parseFloat(e.target.value)})} className="accent-cyan-500 w-full" />
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] text-cyan-300">
                  <span>Phase (phi)</span>
                  <span className="font-mono">{mathSettings.phi.toFixed(2)}</span>
                </div>
                <div className="relative h-1 bg-white/10 rounded-full overflow-hidden mb-1">
                  <motion.div 
                    className="absolute top-0 left-0 h-full bg-cyan-500"
                    animate={{ width: `${(mathSettings.phi / 6.28) * 100}%` }}
                  />
                </div>
                <input type="range" min="0.0" max="6.28" step="0.1" value={mathSettings.phi} onChange={(e) => setMathSettings({...mathSettings, phi: parseFloat(e.target.value)})} className="accent-cyan-500 w-full" />
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[10px] text-cyan-300">
                  <span>Base Amplitude (A)</span>
                  <span className="font-mono">{mathSettings.baseA.toFixed(2)}</span>
                </div>
                <div className="relative h-1 bg-white/10 rounded-full overflow-hidden mb-1">
                  <motion.div 
                    className="absolute top-0 left-0 h-full bg-cyan-500"
                    animate={{ width: `${(mathSettings.baseA / 3.0) * 100}%` }}
                  />
                </div>
                <input type="range" min="0.0" max="3.0" step="0.1" value={mathSettings.baseA} onChange={(e) => setMathSettings({...mathSettings, baseA: parseFloat(e.target.value)})} className="accent-cyan-500 w-full" />
              </div>
            </div>
            
            <div className="text-[10px] text-white/40 mt-2 leading-relaxed">
              Available variables: <br/>
              <span className="text-cyan-400">z</span> (position), <span className="text-cyan-400">t</span> (time), <span className="text-cyan-400">A</span> (amplitude), <span className="text-cyan-400">k</span> (wavenumber), <span className="text-cyan-400">w</span> (angular freq), <span className="text-cyan-400">phi</span> (phase).
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Panel */}
      <AnimatePresence>
        {isSettingsOpen && (
          <motion.div 
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            className="absolute top-0 right-0 w-80 h-full bg-black/90 backdrop-blur-xl z-[150] border-l border-cyan-500/30 p-8 flex flex-col gap-8 shadow-[-20px_0_50px_rgba(0,0,0,0.5)]"
          >
            <div className="flex justify-between items-center">
              <h3 className="text-cyan-400 tracking-widest uppercase font-bold flex items-center gap-2">
                <Sliders size={18} /> Lab Settings
              </h3>
              <button onClick={() => setIsSettingsOpen(false)} className="text-white/50 hover:text-white">
                <X size={20} />
              </button>
            </div>

            <div className="flex flex-col gap-6 overflow-y-auto pr-2">
              {/* Colors */}
              <div className="flex flex-col gap-3">
                <label className="text-[10px] uppercase tracking-widest text-white/40 flex items-center gap-2">
                  <Palette size={12} /> Visual Palette
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col gap-1">
                    <span className="text-[8px] uppercase opacity-50">Base</span>
                    <input 
                      type="color" 
                      value={settings.colorBase} 
                      onChange={(e) => updateSetting('colorBase', e.target.value)}
                      className="w-full h-8 bg-transparent border-none cursor-pointer"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <span className="text-[8px] uppercase opacity-50">Surface</span>
                    <input 
                      type="color" 
                      value={settings.colorSurface} 
                      onChange={(e) => updateSetting('colorSurface', e.target.value)}
                      className="w-full h-8 bg-transparent border-none cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              {/* Simulation */}
              <div className="flex flex-col gap-4">
                <label className="text-[10px] uppercase tracking-widest text-white/40 flex items-center gap-2">
                  <Zap size={12} /> Physics
                </label>
                
                {/* Auto Evolve Toggle */}
                <div className="flex justify-between items-center bg-white/5 p-2 rounded">
                  <span className="text-[10px] uppercase tracking-widest text-cyan-300">Auto-Evolve Waves</span>
                  <button 
                    onClick={() => updateSetting('autoEvolve', !settings.autoEvolve)}
                    className={`px-3 py-1 text-[10px] uppercase tracking-widest rounded transition-all ${settings.autoEvolve ? 'bg-cyan-500 text-black font-bold' : 'bg-gray-800 text-white'}`}
                  >
                    {settings.autoEvolve ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* 3D Waveform Toggle */}
                <div className="flex justify-between items-center bg-white/5 p-2 rounded">
                  <span className="text-[10px] uppercase tracking-widest text-cyan-300">3D Volumetric Waveform</span>
                  <button 
                    onClick={() => updateSetting('show3DWaveform', !settings.show3DWaveform)}
                    className={`px-3 py-1 text-[10px] uppercase tracking-widest rounded transition-all ${settings.show3DWaveform ? 'bg-cyan-500 text-black font-bold' : 'bg-gray-800 text-white'}`}
                  >
                    {settings.show3DWaveform ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* Advanced Rendering */}
                <div className="flex justify-between items-center bg-white/5 p-2 rounded relative group">
                  <span className="text-[10px] uppercase tracking-widest text-cyan-300">Advanced Rendering (SSAO/SSR)</span>
                  <button 
                    onClick={() => updateSetting('enableAdvancedRendering', !settings.enableAdvancedRendering)}
                    className={`px-3 py-1 text-[10px] uppercase tracking-widest rounded transition-all ${settings.enableAdvancedRendering ? 'bg-orange-500 text-black font-bold' : 'bg-gray-800 text-white'}`}
                  >
                    {settings.enableAdvancedRendering ? 'ON' : 'OFF'}
                  </button>
                  <div className="absolute top-full left-0 mt-1 bg-black/90 text-red-400 text-[10px] p-2 rounded border border-red-500/50 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-[200]">
                    Warning: May heavily impact performance in VR.
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex justify-between text-[10px] uppercase">
                    <span>Wave Speed</span>
                    <span className="text-cyan-400">{settings.waveSpeed.toFixed(1)}</span>
                  </div>
                  <input 
                    type="range" min="1" max="30" step="0.5"
                    value={settings.waveSpeed}
                    onChange={(e) => updateSetting('waveSpeed', parseFloat(e.target.value))}
                    className="accent-cyan-500"
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between text-[10px] uppercase">
                    <span>Intensity Falloff</span>
                    <span className="text-cyan-400">{settings.intensityFalloff.toFixed(2)}</span>
                  </div>
                  <input 
                    type="range" min="0.01" max="1.0" step="0.01"
                    value={settings.intensityFalloff}
                    onChange={(e) => updateSetting('intensityFalloff', parseFloat(e.target.value))}
                    className="accent-cyan-500"
                  />
                </div>
              </div>

              {/* Immersion */}
              <div className="flex flex-col gap-4">
                <label className="text-[10px] uppercase tracking-widest text-white/40 flex items-center gap-2">
                  <Volume2 size={12} /> Immersion
                </label>
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between text-[10px] uppercase">
                    <span>Sound Volume</span>
                    <span className="text-cyan-400">{(settings.soundVolume * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" min="0" max="1" step="0.01"
                    value={settings.soundVolume}
                    onChange={(e) => updateSetting('soundVolume', parseFloat(e.target.value))}
                    className="accent-cyan-500"
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between text-[10px] uppercase">
                    <span>Haptic Intensity</span>
                    <span className="text-cyan-400">{(settings.hapticIntensity * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" min="0" max="1" step="0.01"
                    value={settings.hapticIntensity}
                    onChange={(e) => updateSetting('hapticIntensity', parseFloat(e.target.value))}
                    className="accent-cyan-500"
                  />
                </div>
              </div>

              {/* State Management */}
              <div className="flex flex-col gap-4 border-t border-cyan-500/30 pt-4">
                <label className="text-[10px] uppercase tracking-widest text-white/40 flex items-center gap-2">
                  <Save size={12} /> Presets & States
                </label>
                <button 
                  onClick={savePreset}
                  className="w-full py-2 bg-cyan-600/30 hover:bg-cyan-600 text-white text-[10px] tracking-widest uppercase font-bold rounded transition-all"
                >
                  Save Current Configuration
                </button>
                
                {presets.length > 0 && (
                  <div className="flex flex-col gap-2 max-h-32 overflow-y-auto pr-1">
                    {presets.map((p, i) => (
                      <div key={p.id} className="flex justify-between items-center text-[10px] bg-white/5 border border-white/10 p-2 rounded hover:border-cyan-500/50 transition-colors">
                        <span className="truncate max-w-[120px]">{p.name || `State ${i + 1}`}</span>
                        <div className="flex gap-2">
                           <button 
                             onClick={() => loadPreset(p)} 
                             className="text-cyan-400 hover:text-cyan-300 px-2 py-1 bg-cyan-900/40 rounded uppercase tracking-widest transition-colors font-bold"
                           >
                             Load
                           </button>
                           <button 
                             onClick={() => setPresets(presets.filter(pr => pr.id !== p.id))} 
                             className="text-red-400 hover:text-red-300 px-2 py-1 bg-red-900/40 rounded uppercase tracking-widest transition-colors"
                           >
                             Del
                           </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <p className="mt-auto text-[8px] uppercase tracking-widest text-white/20 text-center">
              Cosmic Gel Lab v2.0 // XR Hand Support
            </p>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Top Dropdown Panel UI */}
      <div 
        className="absolute top-0 left-0 w-full h-32 z-50"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <motion.div 
          initial={{ y: '-100%' }}
          animate={{ y: isHovered ? '0%' : '-100%' }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          className="bg-black/80 backdrop-blur-md border-b border-cyan-500/30 p-6 flex flex-col items-center shadow-[0_10px_30px_rgba(0,255,255,0.1)]"
        >
          <h2 className="text-cyan-400 tracking-widest uppercase text-sm mb-4">Select Simulation Layer</h2>
          <div className="flex gap-4">
            {[
              { id: 'fluid', label: 'Cosmic Fluid' },
              { id: 'math', label: 'Mathematical Model' },
              { id: 'hybrid', label: 'Hybrid View' }
            ].map(mode => (
              <button 
                key={mode.id}
                onClick={() => {
                  setViewMode(mode.id)
                  playSound('ui', settings.soundVolume)
                }}
                className={`px-6 py-2 rounded-full text-xs tracking-widest uppercase transition-all ${
                  viewMode === mode.id 
                    ? 'bg-cyan-500 text-black font-bold shadow-[0_0_15px_rgba(34,211,238,0.6)]' 
                    : 'bg-transparent border border-cyan-500/50 text-cyan-500 hover:bg-cyan-500/10'
                }`}
              >
                {mode.label}
              </button>
            ))}
          </div>
        </motion.div>
        
        {/* Hover Indicator */}
        <motion.div 
          animate={{ opacity: isHovered ? 0 : 0.5 }}
          className="absolute top-2 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1"
        >
          <div className="w-12 h-1 bg-cyan-500/50 rounded-full" />
          <span className="text-[8px] uppercase tracking-widest text-cyan-500/50">Hover for Layers</span>
        </motion.div>
      </div>

      {/* Bottom Left Info */}
      <div className="absolute bottom-10 left-10 pointer-events-none select-none max-w-md z-10">
        <h1 className="text-4xl font-thin tracking-tighter uppercase leading-none mb-2">
          {viewMode === 'fluid' && 'Liquid Glass'}
          {viewMode === 'math' && 'Geo3D Math'}
          {viewMode === 'hybrid' && 'Vortex Engine'}
        </h1>
        <div className="h-px w-16 bg-cyan-500 mb-4" />
        <p className="text-xs opacity-60 leading-relaxed text-cyan-100 mb-4">
          {viewMode === 'fluid' && 'A continuous, viscoelastic fluid simulation. Click to create a transverse shear vortex.'}
          {viewMode === 'math' && 'Mathematical representation of the helical wave. Light is a screw-thread, not a flat zigzag.'}
          {viewMode === 'hybrid' && 'Observe the mathematical structure threading through the physical fluid medium.'}
        </p>
      </div>

      <WaveformAnalyzer mathSettings={mathSettings} punchData={punchDataState} />

      <div className="absolute bottom-10 right-10 text-right text-[10px] tracking-widest uppercase pointer-events-none text-cyan-500 opacity-50">
        React Three Fiber / Volumetric Shaders
      </div>
    </div>
  )
}
