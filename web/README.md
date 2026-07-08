# AFR Frontend

This directory contains the web interface for the Automated Feature Recognition application. It is built using React, Vite, Tailwind CSS, and Three.js (via React Three Fiber).

## Overview

The frontend provides an intuitive UI for users to upload their 3D CAD files. It communicates with the backend API to retrieve feature predictions and a 3D mesh. Using `Three.js`, the application renders the mesh interactively in the browser, allowing users to rotate, zoom, and inspect the CAD model and its recognized features.

## Setup Instructions

Make sure you have [Node.js](https://nodejs.org/) installed on your machine.

### 1. Install Dependencies

Navigate to the `web` directory and install the required NPM packages:

```bash
cd web
npm install
```

### 2. Environment Configuration

If you need to point the frontend to a specific backend URL, create a `.env` file in the root of the `web` directory (if it doesn't already exist) and define your API URL. By default, it is configured to communicate with the local backend on `http://localhost:8000`.

### 3. Run the Development Server

Start the Vite development server:

```bash
npm run dev
```

The application will start, typically accessible at `http://localhost:5173`. The terminal will display the exact local URL.

### 4. Build for Production

To create an optimized production build:

```bash
npm run build
```

This will bundle the React application into the `dist/` directory, ready to be served by static file server.
