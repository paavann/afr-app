// 1. Import Bounds and Center from drei
import { Bounds, Center, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useLoader } from '@react-three/fiber'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader'

function MeshComponent({ url }) {
  const geometry = useLoader(STLLoader, url) // This is where it often fails if the URL is wrong
  return <mesh geometry={geometry}><meshStandardMaterial color="orange" /></mesh>
}

export default function Viewport3D({ meshUrl }) {
  return (
    <Canvas camera={{ position: [0, 0, 50], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 10]} intensity={1} />

      {/* 2. Wrap your model in <Bounds> and tell it to auto-fit (fit) and center (clip) */}
      <Bounds fit clip observe margin={1.2}>
        <Center>
          {/* Your STL Loading logic goes here */}
          <mesh>
            {/* ... */}
          </mesh>
        </Center>
      </Bounds>

      {/* 3. Ensure makeDefault is on your controls so the Bounds component can control the camera */}
      <OrbitControls makeDefault />
    </Canvas>
  )
}