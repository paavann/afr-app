import { useRef, useMemo, Component } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows, Float } from '@react-three/drei';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { getFeatureColor, rgbNormalized } from '../utils/colorMap';

/**
 * Error boundary to catch WebGL / Three.js rendering failures.
 */
class WebGLErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full rounded-2xl bg-dark-800 border border-dark-500/20 flex items-center justify-center">
          <div className="text-center p-8 max-w-md">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-red-500/10 flex items-center justify-center">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-red-400">
                <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">WebGL Not Available</h3>
            <p className="text-sm text-dark-200 mb-3">
              3D rendering requires a WebGL-capable browser with GPU acceleration enabled.
            </p>
            <p className="text-xs text-dark-300 font-mono">
              {this.state.error?.message || 'Unknown rendering error'}
            </p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/**
 * Applies per-face vertex colors to an STL geometry based on predictions.
 *
 * STL geometries store faces as individual triangles (non-indexed).
 * Each triangle has 3 vertices, so face i occupies vertices [i*3, i*3+1, i*3+2].
 *
 * When predictions.length < geometry face count, predictions are distributed
 * proportionally across the mesh faces (UV-Net returns per-B-rep-face labels,
 * while STL has per-triangle faces).
 */
function applyFaceColors(geometry, predictions) {
  const positionCount = geometry.attributes.position.count;
  const faceCount = positionCount / 3; // Each STL face = 1 triangle = 3 vertices

  const colors = new Float32Array(positionCount * 3);

  for (let faceIdx = 0; faceIdx < faceCount; faceIdx++) {
    // Map STL triangle index to prediction index
    const predIdx = Math.floor((faceIdx / faceCount) * predictions.length);
    const clampedIdx = Math.min(predIdx, predictions.length - 1);
    const feature = getFeatureColor(predictions[clampedIdx]);
    const [r, g, b] = rgbNormalized(feature.rgb);

    // Set color for all 3 vertices of this face
    for (let v = 0; v < 3; v++) {
      const vertIdx = faceIdx * 3 + v;
      colors[vertIdx * 3]     = r;
      colors[vertIdx * 3 + 1] = g;
      colors[vertIdx * 3 + 2] = b;
    }
  }

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return geometry;
}

/**
 * Generates a procedural demo geometry when no STL file is available.
 * Creates a multi-faced object by merging multiple primitives so each
 * group of faces can be colored by a different prediction label.
 */
function generateMockGeometry(predictions) {
  const geometries = [];
  const predCount = predictions.length;

  // Create a collection of different primitives
  const primitiveGenerators = [
    // Box faces (6 faces × 2 triangles = 12 triangles)
    () => new THREE.BoxGeometry(2, 2, 2, 1, 1, 1),
    // Cylinder (radial segments × 4 triangles roughly)
    () => new THREE.CylinderGeometry(0.8, 0.8, 1.5, 16),
    // Sphere
    () => new THREE.SphereGeometry(0.6, 16, 12),
    // Torus
    () => new THREE.TorusGeometry(0.5, 0.2, 8, 16),
    // Cone
    () => new THREE.ConeGeometry(0.7, 1.2, 12),
  ];

  // Placement positions for sub-geometries
  const positions = [
    [0, 0, 0],
    [2.8, 0.5, 0],
    [-2.5, 0.8, 0.5],
    [0, 1.8, 0],
    [1.5, -1.2, 1],
  ];

  // Create merged geometry
  const mergedGeo = new THREE.BufferGeometry();
  let totalVertices = 0;

  // Calculate how many primitives we need
  const primCount = Math.min(primitiveGenerators.length, 5);

  const allPositions = [];
  const allNormals = [];

  for (let i = 0; i < primCount; i++) {
    const geo = primitiveGenerators[i]();
    const pos = geo.attributes.position;
    const norm = geo.attributes.normal;

    for (let j = 0; j < pos.count; j++) {
      allPositions.push(
        pos.getX(j) + positions[i][0],
        pos.getY(j) + positions[i][1],
        pos.getZ(j) + positions[i][2]
      );
      allNormals.push(norm.getX(j), norm.getY(j), norm.getZ(j));
    }

    // If indexed, we need to de-index
    if (geo.index) {
      const nonIndexed = geo.toNonIndexed();
      const niPos = nonIndexed.attributes.position;
      const niNorm = nonIndexed.attributes.normal;

      // Replace the data we just added
      allPositions.length -= pos.count * 3;
      allNormals.length -= norm.count * 3;

      for (let j = 0; j < niPos.count; j++) {
        allPositions.push(
          niPos.getX(j) + positions[i][0],
          niPos.getY(j) + positions[i][1],
          niPos.getZ(j) + positions[i][2]
        );
        allNormals.push(niNorm.getX(j), niNorm.getY(j), niNorm.getZ(j));
      }
      totalVertices += niPos.count;
      nonIndexed.dispose();
    } else {
      totalVertices += pos.count;
    }

    geo.dispose();
  }

  mergedGeo.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(allPositions), 3)
  );
  mergedGeo.setAttribute(
    'normal',
    new THREE.BufferAttribute(new Float32Array(allNormals), 3)
  );

  // Apply face colors from predictions
  applyFaceColors(mergedGeo, predictions);

  mergedGeo.computeBoundingSphere();
  return mergedGeo;
}


// ──────────────────────────── Three.js Scene Components ────────────────────────────

/**
 * The main 3D mesh component. Handles both STL-loaded and procedural geometry.
 */
function CADMesh({ stlData, predictions }) {
  const meshRef = useRef();

  const geometry = useMemo(() => {
    let geo;

    if (stlData) {
      const loader = new STLLoader();
      geo = loader.parse(stlData);
    } else {
      geo = generateMockGeometry(predictions);
    }

    // Ensure we have computed normals
    geo.computeVertexNormals();

    // Apply per-face vertex colors
    applyFaceColors(geo, predictions);

    // Center geometry
    geo.computeBoundingBox();
    geo.center();

    return geo;
  }, [stlData, predictions]);

  // Gentle rotation
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.1;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        roughness={0.35}
        metalness={0.15}
        envMapIntensity={0.8}
        flatShading
      />
    </mesh>
  );
}

/**
 * Animated grid floor for spatial reference.
 */
function GridFloor() {
  return (
    <group position={[0, -2.5, 0]}>
      <gridHelper
        args={[20, 40, '#2a2a38', '#1a1a26']}
      />
      <ContactShadows
        position={[0, 0.01, 0]}
        opacity={0.35}
        scale={15}
        blur={2.5}
        far={5}
        color="#6c5ce7"
      />
    </group>
  );
}

/**
 * Ambient scene lighting setup.
 */
function SceneLighting() {
  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight
        position={[8, 12, 5]}
        intensity={1.2}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={50}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
      />
      <directionalLight position={[-5, 5, -5]} intensity={0.3} color="#a29bfe" />
      <pointLight position={[0, 8, 0]} intensity={0.5} color="#00d2ff" />
    </>
  );
}


// ──────────────────────────── Main Viewport Component ────────────────────────────

/**
 * Full 3D viewport with Canvas, controls, and environment.
 */
export default function Viewport3D({ stlData, predictions }) {
  return (
    <div className="viewport-canvas w-full h-full rounded-2xl overflow-hidden bg-dark-800 border border-dark-500/20">
      <WebGLErrorBoundary>
        <Canvas
          shadows
          camera={{ position: [5, 4, 6], fov: 45, near: 0.1, far: 200 }}
          gl={{
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance',
            toneMapping: THREE.ACESFilmicToneMapping,
            toneMappingExposure: 1.1,
          }}
          style={{ background: 'linear-gradient(180deg, #12121a 0%, #0a0a0f 100%)' }}
        >
          <SceneLighting />

          {/* Environment for reflections */}
          <Environment preset="city" background={false} />

          {/* The CAD model */}
          <Float speed={0.8} rotationIntensity={0} floatIntensity={0.3}>
            <CADMesh stlData={stlData} predictions={predictions} />
          </Float>

          {/* Floor */}
          <GridFloor />

          {/* Controls */}
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.05}
            minDistance={2}
            maxDistance={30}
            maxPolarAngle={Math.PI / 1.8}
            target={[0, 0, 0]}
          />
        </Canvas>
      </WebGLErrorBoundary>
    </div>
  );
}
