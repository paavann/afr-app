/**
 * Mock data for offline testing.
 *
 * Simulates the backend response from POST /api/predict-cad.
 * The `predictions` array maps face indices to feature class labels.
 */

// Mock predictions — each entry corresponds to a face index in the STL mesh.
// These simulate the UV-Net inference output for a sample mechanical part.
export const MOCK_PREDICTIONS = [
  'Extrude Side', 'Extrude Side', 'Extrude Side', 'Extrude Side', 'Extrude Side', 'Extrude Side',
  'Extrude End', 'Extrude End', 'Extrude End', 'Extrude End',
  'Fillet', 'Fillet', 'Fillet', 'Fillet', 'Fillet', 'Fillet',
  'Fillet', 'Fillet',
  'Chamfer', 'Chamfer', 'Chamfer',
  'Cut Side', 'Cut Side',
  'Cut End', 'Cut End',
  'Revolve',
  'Stock', 'Stock', 'Stock', 'Stock',
  'Other', 'Other',
  'Extrude Side',
  'Extrude End', 'Cut Side',
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
