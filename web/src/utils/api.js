import axios from 'axios';

const API_BASE = import.meta.env.VITE_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120_000, // 2 minutes for large CAD files
});

/**
 * Upload a .STEP file for CAD feature prediction.
 *
 * @param {File} file — The .step/.stp file to upload
 * @param {function} onProgress — Progress callback (0–100)
 * @returns {Promise<{ predictions: string[], file_info: object, stl_data: ArrayBuffer }>}
 */
export async function predictCAD(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/api/afr', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    responseType: 'json',
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total) {
        const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
        onProgress?.(percent);
      }
    },
  });

  return response.data;
}

/**
 * Download the processed STL mesh from the backend.
 *
 * @param {string} stlUrl — Relative or absolute URL to the STL file
 * @returns {Promise<ArrayBuffer>}
 */
export async function fetchSTL(stlUrl) {
  let url = stlUrl.startsWith('http') ? stlUrl : `${API_BASE}${stlUrl}`;

  // Cloud Proxy Fix: Force HTTPS if the URL belongs to Hugging Face
  // (We use an IF statement so we don't break your localhost testing later)
  if (url.includes('hf.space')) {
    url = url.replace('http://', 'https://');
  }

  const response = await axios.get(url, {
    responseType: 'arraybuffer',
    timeout: 60_000,
  });

  return response.data;
}
