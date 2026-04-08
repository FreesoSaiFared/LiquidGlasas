import React, { useRef, useMemo, useState, useCallback } from 'react'
import { Canvas, useFrame, ThreeEvent, useThree, extend, ThreeElement } from '@react-three/fiber'
import { OrbitControls, Stars, PerspectiveCamera, shaderMaterial, Text } from '@react-three/drei'
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import * as THREE from 'three'
import { motion, AnimatePresence } from 'motion/react'

// --- 1. True Fluid Shader (Liquid Glass) ---
const FluidGelMaterial = shaderMaterial(
  {
    uTime: 0,
    uPunchPos: new THREE.Vector3(0, 0, 0),
    uPunchTime: -100.0,
    uColorBase: new THREE.Color('#000a1f'), // Very dark deep blue
    uColorSurface: new THREE.Color('#00ffff'), // Bright cyan for reflections
  },
  // Vertex Shader: Fluid Deformation
  `
    varying vec3 vPos;
    varying vec3 vViewPosition;
    varying vec2 vUv;
    uniform float uTime;
    uniform vec3 uPunchPos;
    uniform float uPunchTime;

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
         float waveSpeed = 12.0;
         float waveFront = t * waveSpeed;
         float distToWave = abs(dist - waveFront);
         
         // Gaussian falloff for the ripple
         float intensity = exp(-distToWave * 0.2) * exp(-t * 0.15);
         
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

extend({ FluidGelMaterial })

declare global {
  namespace JSX {
    interface IntrinsicElements {
      fluidGelMaterial: ThreeElement<typeof FluidGelMaterial>
    }
  }
}

// --- 2. The Continuous Fluid Component ---
const FluidGelMedium = ({ punchData }: { punchData: { position: THREE.Vector3; time: number } | null }) => {
  const materialRef = useRef<any>(null!)

  useFrame(({ clock }) => {
    if (materialRef.current) {
      materialRef.current.uTime = clock.getElapsedTime()
      if (punchData) {
        materialRef.current.uPunchPos.copy(punchData.position)
        materialRef.current.uPunchTime = punchData.time
      }
    }
  })

  return (
    <mesh>
      {/* High resolution icosahedron for uniform fluid vertex deformation (removes the grid pattern) */}
      <icosahedronGeometry args={[80, 200]} />
      {/* @ts-ignore */}
      <fluidGelMaterial 
        ref={materialRef} 
        transparent={true} 
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

// --- 3. The Mathematical 3D Graph (Geo3D Style) ---
const MathGraph = ({ punchData }: { punchData: { position: THREE.Vector3; time: number } | null }) => {
  const pointsCount = 3000;
  const helix1Ref = useRef<THREE.Line>(null!);
  const helix2Ref = useRef<THREE.Line>(null!);
  const groupRef = useRef<THREE.Group>(null!);

  const positions1 = useMemo(() => new Float32Array(pointsCount * 3), []);
  const positions2 = useMemo(() => new Float32Array(pointsCount * 3), []);

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
      
      for (let i = 0; i < pointsCount; i++) {
        const z = (i / pointsCount) * 160 - 80; // Spread along Z axis from -80 to 80
        
        // Calculate intensity based on the wave disturbance
        let intensity = 0.05; // Base faint helix
        if (punchData) {
          const dist = Math.abs(80 - z); // Distance from punch (which is at z=80)
          const tPunch = t - punchData.time;
          if (tPunch > 0 && tPunch < 20.0) {
            const waveFront = tPunch * 12.0;
            const distToWave = Math.abs(dist - waveFront);
            const pulse = Math.exp(-distToWave * 0.2) * Math.exp(-tPunch * 0.15);
            intensity += pulse;
          }
        }

        const amplitude = 8.0 * intensity;
        
        // The mathematical wave propagates
        const phase = z * 0.3 - t * 3.0;
        
        // Helix 1 (e.g., Electric Field equivalent)
        pos1[i * 3] = Math.cos(phase) * amplitude;
        pos1[i * 3 + 1] = Math.sin(phase) * amplitude;
        pos1[i * 3 + 2] = z;

        // Helix 2 (e.g., Magnetic Field equivalent, offset by PI)
        pos2[i * 3] = Math.cos(phase + Math.PI) * amplitude;
        pos2[i * 3 + 1] = Math.sin(phase + Math.PI) * amplitude;
        pos2[i * 3 + 2] = z;
      }
      
      helix1Ref.current.geometry.attributes.position.needsUpdate = true;
      helix2Ref.current.geometry.attributes.position.needsUpdate = true;
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
        <lineBasicMaterial color="#00ffff" linewidth={2} />
      </line>
      {/* @ts-ignore */}
      <line ref={helix2Ref}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={pointsCount} array={positions2} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#ff00ff" linewidth={2} />
      </line>

      {/* Mathematical Labels */}
      <Text position={[16, 0, 0]} fontSize={1.5} color="#ff0055" anchorX="left">X (Transverse)</Text>
      <Text position={[0, 16, 0]} fontSize={1.5} color="#00ff55" anchorX="center">Y (Transverse)</Text>
      <Text position={[0, 0, 40]} fontSize={1.5} color="#ffffff" anchorX="left">Z (Longitudinal Propagation)</Text>
      <Text position={[10, 10, 0]} fontSize={2} color="#00ffff" rotation={[0, -Math.PI/4, 0]}>Helical Vortex Wave</Text>
    </group>
  )
}

// --- 4. The Scene Controller ---
const Scene = ({ viewMode }: { viewMode: string }) => {
  const [punchData, setPunchData] = useState<{ position: THREE.Vector3; time: number } | null>(null)
  const { clock } = useThree()

  const handlePunch = useCallback((event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    setPunchData({
      position: event.point,
      time: clock.getElapsedTime()
    })
  }, [clock])

  return (
    <>
      <OrbitControls makeDefault minDistance={5} maxDistance={60} />
      
      {/* Background Space */}
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

      <group onPointerDown={handlePunch}>
        {/* Render layers based on dropdown selection */}
        {(viewMode === 'fluid' || viewMode === 'hybrid') && <FluidGelMedium punchData={punchData} />}
        {(viewMode === 'math' || viewMode === 'hybrid') && <MathGraph punchData={punchData} />}
        
        {/* Invisible sphere to catch clicks across the volume */}
        <mesh>
          <sphereGeometry args={[80, 32, 32]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      </group>

      {/* Post-processing for Dazzling Visuals */}
      <EffectComposer>
        <Bloom 
          luminanceThreshold={0.2} 
          luminanceSmoothing={0.9} 
          intensity={1.5} 
          mipmapBlur 
        />
        <ChromaticAberration 
          blendFunction={BlendFunction.NORMAL} 
          offset={new THREE.Vector2(0.002, 0.002)} 
        />
      </EffectComposer>
    </>
  )
}

// --- 5. The Main App with UI ---
export default function App() {
  const [isHovered, setIsHovered] = useState(false)
  const [viewMode, setViewMode] = useState('fluid') // 'fluid', 'math', 'hybrid'

  return (
    <div className="w-screen h-screen bg-[#000000] overflow-hidden relative font-sans text-white">
      <Canvas camera={{ position: [0, 10, 30], fov: 45 }} dpr={[1, 2]}>
        <color attach="background" args={['#000000']} />
        <Scene viewMode={viewMode} />
      </Canvas>
      
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
                onClick={() => setViewMode(mode.id)}
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
        <p className="text-xs opacity-60 leading-relaxed text-cyan-100">
          {viewMode === 'fluid' && 'A continuous, viscoelastic fluid simulation. Click to create a transverse shear vortex.'}
          {viewMode === 'math' && 'Mathematical representation of the helical wave. Light is a screw-thread, not a flat zigzag.'}
          {viewMode === 'hybrid' && 'Observe the mathematical structure threading through the physical fluid medium.'}
        </p>
      </div>

      <div className="absolute bottom-10 right-10 text-right text-[10px] tracking-widest uppercase pointer-events-none text-cyan-500 opacity-50">
        React Three Fiber / Volumetric Shaders
      </div>
    </div>
  )
}
