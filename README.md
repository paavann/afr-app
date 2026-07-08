# Automated Feature Recognition (AFR) App

Welcome to the Automated Feature Recognition (AFR) App project. This application provides a full-stack solution for segmenting and recognizing features on 3D CAD models using deep learning.

## Overview

The AFR App allows users to upload standard 3D CAD files (like `.step` or `.stp`) through a sleek web interface. In the background, the geometry is parsed, converted into a mathematical graph representation, and passed through a neural network (UV-Net) to identify specific machining features such as slots, holes, and pockets on the 3D model.

The repository is divided into two main components:

1. **Backend (`/api`)**: A robust Python FastAPI server that handles CAD file parsing using `pythonocc-core`, graph construction using `DGL` (Deep Graph Library), and machine learning inference. It provides RESTful endpoints to process CAD files and return feature predictions along with a rendered 3D mesh.
2. **Frontend (`/web`)**: A modern React application built with Vite and Tailwind CSS. It leverages `Three.js` (via `@react-three/fiber`) to render 3D meshes in the browser and visualize the detected features interactively.

## Navigation

- **`/api`**: Contains all the backend logic, machine learning models, and CAD processing pipelines. Navigate here for API setup and development.
- **`/web`**: Contains the frontend user interface, 3D visualization components, and styling. Navigate here for UI development.

For detailed instructions on setting up each component, please refer to the respective `README.md` files located in their directories:
- [Backend Documentation](./api/README.md)
- [Frontend Documentation](./web/README.md)
