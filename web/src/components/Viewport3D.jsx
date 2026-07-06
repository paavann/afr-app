import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Bounds, Center, GizmoHelper, GizmoViewport } from '@react-three/drei';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { getFeatureColor } from '../utils/colorMap';

/**
 * Renders a colored STL mesh using vertex colors derived from predictions + face mapping.
 */
function ColoredSTLMesh({ stlData, predictions, faceMapping }) {
  const meshData = useMemo(() => {
    const loader = new STLLoader();
    const geom = loader.parse(stlData);
    geom.computeVertexNormals();

    const vertCount = geom.attributes.position.count;
    const triCount = vertCount / 3;

    if (faceMapping && predictions && predictions.length > 0) {
      const colors = new Float32Array(vertCount * 3);

      // Build face_id → type lookup
      const faceTypeMap = {};
      for (const pred of predictions) {
        faceTypeMap[pred.face_id] = pred.type;
      }

      for (let tri = 0; tri < triCount; tri++) {
        const faceId = tri < faceMapping.length ? faceMapping[tri] : 0;
        const featureType = faceTypeMap[faceId] || 'Other';
        const colorInfo = getFeatureColor(featureType);
        const r = colorInfo.rgb[0] / 255;
        const g = colorInfo.rgb[1] / 255;
        const b = colorInfo.rgb[2] / 255;

        for (let v = 0; v < 3; v++) {
          const idx = (tri * 3 + v) * 3;
          colors[idx] = r;
          colors[idx + 1] = g;
          colors[idx + 2] = b;
        }
      }

      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      return {
        geometry: geom,
        material: new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.35,
          metalness: 0.15,
          flatShading: false,
        }),
      };
    }

    // No face mapping — render with a nice solid material
    return {
      geometry: geom,
      material: new THREE.MeshStandardMaterial({
        color: '#6c5ce7',
        roughness: 0.3,
        metalness: 0.2,
        flatShading: false,
      }),
    };
  }, [stlData, predictions, faceMapping]);

  return <mesh geometry={meshData.geometry} material={meshData.material} />;
}


/**
 * Procedural mock geometry for demo mode — colored shapes representing a mechanical part.
 */
function MockGeometry({ predictions }) {
  const group = useMemo(() => {
    const g = new THREE.Group();
    const predLabels = predictions || [];

    // Assign predictions to shapes (or use defaults)
    const getColorForIndex = (idx) => {
      if (idx < predLabels.length) {
        const label = typeof predLabels[idx] === 'string' ? predLabels[idx] : predLabels[idx]?.type || 'Other';
        const c = getFeatureColor(label);
        return new THREE.Color(c.rgb[0] / 255, c.rgb[1] / 255, c.rgb[2] / 255);
      }
      return new THREE.Color(0.4, 0.4, 0.5);
    };

    let predIdx = 0;

    // Base plate — "Plane" faces
    const baseMat = new THREE.MeshStandardMaterial({ color: getColorForIndex(predIdx++), roughness: 0.4, metalness: 0.1 });
    for (let i = 0; i < 5; i++) predIdx++; // Skip plane predictions
    const baseGeom = new THREE.BoxGeometry(4, 0.4, 3);
    const baseMesh = new THREE.Mesh(baseGeom, baseMat);
    baseMesh.position.set(0, -0.2, 0);
    g.add(baseMesh);

    // Cylinders — "Cylinder" faces
    const cylMat = new THREE.MeshStandardMaterial({ color: getColorForIndex(predIdx++), roughness: 0.3, metalness: 0.25 });
    for (let i = 0; i < 3; i++) predIdx++;
    const cylGeom = new THREE.CylinderGeometry(0.3, 0.3, 1.2, 32);
    const cyl1 = new THREE.Mesh(cylGeom, cylMat);
    cyl1.position.set(-1.2, 0.6, 0.5);
    g.add(cyl1);
    const cyl2 = new THREE.Mesh(cylGeom.clone(), cylMat);
    cyl2.position.set(1.2, 0.6, -0.5);
    g.add(cyl2);

    // Fillets — "Fillet" faces (torus sections)
    const filletMat = new THREE.MeshStandardMaterial({ color: getColorForIndex(predIdx++), roughness: 0.3, metalness: 0.1 });
    for (let i = 0; i < 7; i++) predIdx++;
    const filletGeom = new THREE.TorusGeometry(0.35, 0.08, 16, 32, Math.PI / 2);
    const positions = [
      [-2, 0, -1.5], [2, 0, -1.5], [-2, 0, 1.5], [2, 0, 1.5],
    ];
    const rotations = [
      [0, 0, 0], [0, Math.PI / 2, 0], [0, -Math.PI / 2, 0], [0, Math.PI, 0],
    ];
    positions.forEach((pos, i) => {
      const fillet = new THREE.Mesh(filletGeom.clone(), filletMat);
      fillet.position.set(...pos);
      if (rotations[i]) fillet.rotation.set(...rotations[i]);
      g.add(fillet);
    });

    // Chamfers — beveled edges
    const chamferMat = new THREE.MeshStandardMaterial({ color: getColorForIndex(predIdx++), roughness: 0.4, metalness: 0.15 });
    for (let i = 0; i < 2; i++) predIdx++;
    const chamferGeom = new THREE.BoxGeometry(0.15, 0.15, 3);
    const chamfer1 = new THREE.Mesh(chamferGeom, chamferMat);
    chamfer1.position.set(2.05, 0.2, 0);
    chamfer1.rotation.set(0, 0, Math.PI / 4);
    g.add(chamfer1);

    // Cone
    const coneMat = new THREE.MeshStandardMaterial({ color: getColorForIndex(predIdx++), roughness: 0.35, metalness: 0.2 });
    predIdx++;
    const coneGeom = new THREE.ConeGeometry(0.25, 0.6, 32);
    const cone = new THREE.Mesh(coneGeom, coneMat);
    cone.position.set(0, 0.5, 0);
    g.add(cone);

    // Sphere
    const sphereMat = new THREE.MeshStandardMaterial({ color: getColorForIndex(predIdx++), roughness: 0.25, metalness: 0.3 });
    const sphereGeom = new THREE.SphereGeometry(0.2, 32, 32);
    const sphere = new THREE.Mesh(sphereGeom, sphereMat);
    sphere.position.set(0.6, 0.4, 0.8);
    g.add(sphere);

    return g;
  }, [predictions]);

  return <primitive object={group} />;
}


/**
 * 3D viewport for CAD model visualization.
 * Renders STL meshes with per-face coloring from UV-Net predictions.
 */
export default function Viewport3D({ stlData, predictions, faceMapping }) {
  return (
    <div className="viewport-canvas w-full h-full rounded-none overflow-hidden" style={{ minHeight: '500px', background: 'linear-gradient(135deg, #0a0a0f 0%, #12121a 50%, #0d0d14 100%)' }}>
      <Canvas
        camera={{ position: [3, 2, 5], fov: 45 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        onCreated={({ gl }) => { gl.setClearColor('#0c0c14'); gl.toneMapping = THREE.ACESFilmicToneMapping; gl.toneMappingExposure = 1.2; }}
      >
        {/* Lighting setup */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[8, 10, 6]} intensity={1.4} castShadow color="#ffffff" />
        <directionalLight position={[-5, 5, -5]} intensity={0.4} color="#6c5ce7" />
        <directionalLight position={[0, -3, 5]} intensity={0.2} color="#00d2ff" />

        {/* Auto-fit and center the model */}
        <Bounds fit clip observe margin={1.6}>
          <Center>
            {stlData ? (
              <ColoredSTLMesh stlData={stlData} predictions={predictions} faceMapping={faceMapping} />
            ) : (
              <MockGeometry predictions={predictions} />
            )}
          </Center>
        </Bounds>

        {/* Subtle ground grid */}
        <gridHelper args={[20, 40, '#1a1a26', '#1a1a26']} position={[0, -1.5, 0]} />

        {/* Controls */}
        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.6}
          zoomSpeed={0.8}
          minDistance={1}
          maxDistance={100}
        />

        {/* Orientation gizmo */}
        <GizmoHelper alignment="bottom-left" margin={[60, 60]}>
          <GizmoViewport axisColors={['#e17055', '#00b894', '#6c5ce7']} labelColor="white" />
        </GizmoHelper>
      </Canvas>
    </div>
  );
}