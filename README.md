Offline-ready, privacy-first application that leverages the most suitable hardware available on the edge to run AI workloads.

![Logo](https://github.com/webmaxru/nextjs-webnn/raw/main/public/icons/icon-512x512.png)

## Features

- Feature detection: WebGPU, WebNN, NPU
- Running AI inference in the worker thread, off the main thread
- Installable, offline-ready PWA

## Technologies used

- Nextjs (static mode)
- Workbox: offline-readiness, precaching, runtime caching, smart update flow
- Transformers.js: AI pipelines, models caching. Using ONNX Web Runtime and WebNN, WebGPU under the hood

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.js`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
