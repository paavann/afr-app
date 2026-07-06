/**
 * CAD Feature → Color mapping utility.
 *
 * Maps face classification labels (as returned by the UV-Net inference backend)
 * to distinct HSL colors for 3D rendering.
 */

export const FEATURE_COLORS = {
  // Canonical CAD features
  'Fillet':                { hex: '#6c5ce7', rgb: [108, 92,  231], label: 'Fillet' },
  'Chamfer':               { hex: '#e17055', rgb: [225, 112, 85],  label: 'Chamfer' },
  'Plane':                 { hex: '#636e72', rgb: [99,  110, 114], label: 'Planar Face' },
  'Cylinder':              { hex: '#00b894', rgb: [0,   184, 148], label: 'Cylindrical' },
  'Cone':                  { hex: '#fdcb6e', rgb: [253, 203, 110], label: 'Conical' },
  'Sphere':                { hex: '#00cec9', rgb: [0,   206, 201], label: 'Spherical' },
  'Torus':                 { hex: '#e84393', rgb: [232, 67,  147], label: 'Toroidal' },
  'Spline':                { hex: '#0984e3', rgb: [9,   132, 227], label: 'Spline / NURBS' },
  'Revolution':            { hex: '#fd79a8', rgb: [253, 121, 168], label: 'Revolution' },
  'Extrusion':             { hex: '#55efc4', rgb: [85,  239, 196], label: 'Extrusion' },
  'BSpline':               { hex: '#74b9ff', rgb: [116, 185, 255], label: 'B-Spline' },
  'Other':                 { hex: '#dfe6e9', rgb: [223, 230, 233], label: 'Other / Unknown' },
};

/**
 * Extended palette for unknown numeric class IDs beyond the named set.
 * Uses perceptually distinct colors via golden-angle hue spacing.
 */
const EXTENDED_PALETTE = [
  '#ff6b6b', '#48dbfb', '#feca57', '#ff9ff3', '#54a0ff',
  '#5f27cd', '#01a3a4', '#c44569', '#f8b739', '#3dc1d3',
  '#e15f41', '#4834d4', '#22a6b3', '#6ab04c', '#eb4d4b',
];

/**
 * Map a label string or numeric index to a feature color entry.
 *
 * @param {string|number} classId — e.g. "Fillet", "Chamfer", or numeric index
 * @returns {{ hex: string, rgb: number[], label: string }}
 */
export function getFeatureColor(classId) {
  // String lookup (case-insensitive)
  if (typeof classId === 'string') {
    // Direct match first
    const directMatch = FEATURE_COLORS[classId];
    if (directMatch) return directMatch;

    // Case-insensitive match
    const normalizedId = classId.toLowerCase();
    for (const [key, value] of Object.entries(FEATURE_COLORS)) {
      if (key.toLowerCase() === normalizedId) return value;
    }

    // Try matching partial / underscore-separated names (e.g., 'rectangular_through_slot')
    const readable = classId.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const partialMatch = FEATURE_COLORS[readable];
    if (partialMatch) return partialMatch;

    // If unrecognized, pick a consistent color from EXTENDED_PALETTE based on string hash
    let hash = 0;
    for (let i = 0; i < classId.length; i++) {
      hash = classId.charCodeAt(i) + ((hash << 5) - hash);
    }
    const extIdx = Math.abs(hash) % EXTENDED_PALETTE.length;
    const hex = EXTENDED_PALETTE[extIdx];
    return {
      hex,
      rgb: hexToRgb(hex),
      label: classId.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    };
  }

  // Numeric index → ordered feature keys
  if (typeof classId === 'number') {
    const keys = Object.keys(FEATURE_COLORS);
    if (classId >= 0 && classId < keys.length) {
      return FEATURE_COLORS[keys[classId]];
    }
    // Fallback to extended palette
    const extIdx = classId % EXTENDED_PALETTE.length;
    const hex = EXTENDED_PALETTE[extIdx];
    return {
      hex,
      rgb: hexToRgb(hex),
      label: `Class ${classId}`,
    };
  }

  return FEATURE_COLORS['Other'];
}

/**
 * Get all unique features detected in a predictions array.
 *
 * @param {(string|number)[]} predictions
 * @returns {{ hex: string, rgb: number[], label: string, count: number }[]}
 */
export function getDetectedFeatures(predictions) {
  const countMap = new Map();

  for (const pred of predictions) {
    const color = getFeatureColor(pred);
    const key = color.label;
    if (countMap.has(key)) {
      countMap.get(key).count++;
    } else {
      countMap.set(key, { ...color, count: 1 });
    }
  }

  return Array.from(countMap.values()).sort((a, b) => b.count - a.count);
}

/**
 * Convert a hex color string to an [r, g, b] tuple (0–255).
 */
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [223, 230, 233];
}

/**
 * Normalize an [r,g,b] tuple (0–255) to [0,1] range for Three.js.
 */
export function rgbNormalized(rgb) {
  return rgb.map(c => c / 255);
}
