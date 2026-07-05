/**
 * Mock data for offline testing.
 *
 * Simulates the backend response from POST /api/predict-cad.
 * The `predictions` array maps face indices to feature class labels.
 */

// Mock predictions — each entry corresponds to a face index in the STL mesh.
// These simulate the UV-Net inference output for a sample mechanical part.
export const MOCK_PREDICTIONS = [
  'Plane', 'Plane', 'Plane', 'Plane', 'Plane', 'Plane',
  'Cylinder', 'Cylinder', 'Cylinder', 'Cylinder',
  'Fillet', 'Fillet', 'Fillet', 'Fillet', 'Fillet', 'Fillet',
  'Fillet', 'Fillet',
  'Chamfer', 'Chamfer', 'Chamfer',
  'Cone', 'Cone',
  'Sphere', 'Sphere',
  'Torus',
  'Extrusion', 'Extrusion', 'Extrusion', 'Extrusion',
  'BSpline', 'BSpline',
  'Revolution',
  'Spline', 'Spline',
  'Other',
];

/**
 * Mock metadata about the processed CAD file.
 */
export const MOCK_FILE_INFO = {
  filename: 'sample_bracket.step',
  num_faces: 36,
  num_vertices: 2184,
  num_triangles: 4368,
  processing_time_ms: 847,
  model_version: 'uvnet-v2.1',
  confidence: 0.94,
};

/**
 * Returns the full mock API response shape.
 */
export function getMockResponse() {
  return {
    success: true,
    data: {
      predictions: MOCK_PREDICTIONS,
      file_info: MOCK_FILE_INFO,
      stl_url: null, // No actual STL in mock mode; geometry is procedurally generated
    },
  };
}
