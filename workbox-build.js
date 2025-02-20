const { injectManifest } = require("workbox-build");

let workboxConfig = {
  globDirectory: "out",
  globPatterns: [
    "index.html",
    "*.css",
    "*.js",
    "_next/**/*",
    "samples/**/*",
    "icons/**/*",
    "wasm/**/*",
  ],
  globIgnores: [
    // Skip ES5 bundles
    "**/*-es5.*.js",
  ],

  swSrc: "src/service-worker.js",
  swDest: "out/sw.js",

  // Framework takes care of cache busting for JS and CSS (in prod mode)
  dontCacheBustURLsMatching: new RegExp(".+.[a-f0-9]{20}.(?:js|css)"),

  // By default, Workbox will not cache files larger than 2Mb (might be an issue for dev builds)
  maximumFileSizeToCacheInBytes: 4 * 1024 * 1024, // 4Mb
};

injectManifest(workboxConfig).then(({ count, size }) => {
  console.log(
    `Generated ${workboxConfig.swDest}, which will precache ${count} files, totaling ${size} bytes.`
  );
});
