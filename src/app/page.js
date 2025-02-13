'use client';

import { useState, useEffect, useRef, useCallback, use } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Workbox } from 'workbox-window';

const sampleImages = [
  { name: 'Sample 1', url: '/samples/plane.jpg' },
  { name: 'Sample 2', url: '/samples/cats.jpg' },
  { name: 'Sample 3', url: '/samples/staircase.jpg' },
  { name: 'Sample 4', url: '/samples/desk.jpg' },
  { name: 'Upload file', url: null },
];

async function isWebNN() {
  let navigatorObj = navigator;

  if (typeof MLGraphBuilder !== 'undefined') {
    const context = await navigatorObj.ml.createContext();
    return !context.tf;
  } else {
    return false;
  }
}

async function isWebGPU() {
  let navigatorObj = navigator;

  if (navigatorObj.gpu) {
    try {
      await navigatorObj.gpu.requestAdapter();
      return true;
    } catch (e) {
      return false;
    }
  } else {
    return false;
  }
}

async function isNPU() {
  let navigatorObj = navigator;

  try {
    await navigatorObj.ml.createContext({ deviceType: 'npu' });
    return true;
  } catch (e) {
    return false;
  }
}

async function isFp16() {
  let navigatorObj = navigator;

  try {
    const adapter = await navigatorObj.gpu.requestAdapter();
    return adapter.features.has('shader-f16');
  } catch (e) {
    return false;
  }
}

async function isWorker() {
  if (typeof Worker !== 'undefined') {
    return true;
  } else {
    return false;
  }
}

export default function Home() {
  const [selectedTab, setSelectedTab] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState('webgpu');
  const [selectedImageUrl, setSelectedImageUrl] = useState(sampleImages[0].url);
  const [selectedWebNNDevice, setSelectedWebNNDevice] = useState('');
  const [selectedQuantization, setSelectedQuantization] = useState('fp32');
  const [selectedModel, setSelectedModel] = useState(
    'Xenova/fastvit_t12.apple_in1k'
  );

  const [isWebGPUEnabled, setIsWebGPUEnabled] = useState(true);
  const [isWebNNEnabled, setIsWebNNEnabled] = useState(true);
  const [isNPUEnabled, setIsNPUEnabled] = useState(false);
  const [isFp16Enabled, setIsFp16Enabled] = useState(true);
  const [isWorkerEnabled, setIsWorkerEnabled] = useState(true);

  // Create a reference to the worker object.
  const worker = useRef(null);

  const classify = useCallback(() => {
    if (worker.current) {
      worker.current.postMessage({
        model: selectedModel,
        device:
          selectedDevice == 'webnn'
            ? [selectedDevice, selectedWebNNDevice].filter(Boolean).join('-')
            : selectedDevice,
        dtype: selectedQuantization,
        task: 'image-classification',
        input: selectedImageUrl,
      });

      setLoading(true);
    }
  }, [
    selectedImageUrl,
    selectedDevice,
    selectedWebNNDevice,
    selectedQuantization,
    selectedModel,
  ]);

  const onTabChange = (index) => {
    setSelectedTab(index);
    setSelectedImageUrl(sampleImages[index].url);
    if (index === sampleImages.length - 1) {
      setSelectedImageUrl(null);
    }
    if (result) {
      setResult(null);
      setLoading(false);
    }
  };

  // We use the `useEffect` hook to set up the worker as soon as the `App` component is mounted.
  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL('./worker.js', import.meta.url), {
        type: 'module',
      });
    }

    // Doing feature detection.
    (async () => {
      setIsWebGPUEnabled(await isWebGPU());
      setIsWebNNEnabled(await isWebNN());
      setIsNPUEnabled(await isNPU());
      setIsFp16Enabled(await isFp16());
      setIsWorkerEnabled(await isWorker());
    })();

    if ('serviceWorker' in navigator) {
      const wb = new Workbox('/sw.js');

      const refreshPage = () => {
        wb.addEventListener('controlling', (event) => {
          window.location.reload();
        });

        wb.messageSkipWaiting();
      };

      const Msg = () => (
        <div>
          Updated app is available&nbsp;&nbsp;
          <button onClick={refreshPage}>Reload</button>
        </div>
      );

      const showSkipWaitingPrompt = (event) => {
        toast.info(<Msg />);
      };

      // Add an event listener to detect when the registered
      // service worker has installed but is waiting to activate.
      wb.addEventListener('waiting', showSkipWaitingPrompt);

      wb.register()
        .then((reg) => {
          console.log('Successful service worker registration', reg);
        })
        .catch((err) =>
          console.error('Service worker registration failed', err)
        );
    } else {
      console.error('Service Worker API is not supported in current browser');
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case 'initiate':
          setReady(false);
          break;
        case 'ready':
          setReady(true);
          break;
        case 'complete':
          setResult(e.data.output);
          setReady(true);
          setLoading(false);
          break;
      }
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener('message', onMessageReceived);

    // Define a cleanup function for when the component is unmounted.
    return () =>
      worker.current.removeEventListener('message', onMessageReceived);
  }, []);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = (e) => {
        setSelectedImageUrl(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      alert('Only image files are supported!');
    }
  };

  return (
    <>
      <header className="w-full p-3 px-6 text-center">
        <h1 className="text-4xl font-bold">WebNN using Transformers.js</h1>
        <p className="text-lg">
          Made in Norway by&nbsp;
          <a className="underline" href="https://www.linkedin.com/in/webmax/">
            Maxim Salnikov
          </a>
          &nbsp;|&nbsp;
          <a
            className="underline"
            href="https://github.com/webmaxru/nextjs-webnn"
          >
            GitHub Repo
          </a>
          &nbsp;|&nbsp;
          <a
            className="underline"
            href="https://sessionize.com/s/maxim-salnikov/privacy-first-in-browser-generative-ai-web-apps-of/105248"
          >
            Tech talk
          </a>
          &nbsp;|&nbsp;
          <a
            className="underline"
            href="https://www.slideshare.net/slideshow/privacy-first-in-browser-generative-ai-web-apps-offline-ready-future-proof-standards-based/273142915"
          >
            Slides
          </a>
        </p>
      </header>

      <main className="flex flex-col w-full p-3 px-6 gap-y-3">
        <h2 className="text-2xl font-bold">
          Image Classification: Assigning a label or class to an entire image
        </h2>

        <div className="flex flex-row gap-4 gap-y-0 flex-wrap content-end items-end">
          <ModelSelector
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
          />

          <APISelector
            selectedDevice={selectedDevice}
            setSelectedDevice={setSelectedDevice}
            isWebGPUEnabled={isWebGPUEnabled}
            isWebNNEnabled={isWebNNEnabled}
          />
          <WebNNDeviceSelector
            selectedWebNNDevice={selectedWebNNDevice}
            setSelectedWebNNDevice={setSelectedWebNNDevice}
            isWebNNSelected={selectedDevice == 'webnn'}
            isNPUEnabled={isNPUEnabled}
          />
          <QuantizationSelector
            selectedQuantization={selectedQuantization}
            setSelectedQuantization={setSelectedQuantization}
            isFp16Enabled={isFp16Enabled}
          />

          <div className="label flex flex-col items-start gap-2">
            <button
              className="btn btn-accent"
              disabled={loading || !isWorkerEnabled}
              onClick={classify}
            >
              {loading && <span className="loading loading-spinner"></span>}
              Classify
            </button>
          </div>
        </div>

        <Output
          result={result}
          loading={loading}
          selectedDevice={selectedDevice}
        />

        <SampleTabs
          selectedTab={selectedTab}
          setSelectedTab={onTabChange}
          handleFileUpload={handleFileUpload}
          selectedImageUrl={selectedImageUrl}
        />
      </main>
    </>
  );
}

// Output Component
const Output = ({ result, loading, selectedDevice }) => {
  if (!result && !loading) return null;

  return (
    <>
      <h2 className="text-2xl font-bold">Results:</h2>
      <div className="p-6 w-full rounded-xl bg-base-200 overflow-x-scroll">
        {loading ? (
          <span className="loading loading-dots loading-lg"></span>
        ) : (
          result?.map((item, index) => (
            <pre key={index} data-prefix="$">
              Label: {item.label}, Score: {item.score.toFixed(2)}
            </pre>
          ))
        )}
      </div>
    </>
  );
};

// API Control Component
const APISelector = ({
  selectedDevice,
  setSelectedDevice,
  isWebGPUEnabled,
  isWebNNEnabled,
}) => {
  const options = [
    { label: 'Auto', value: 'auto', disabled: false },
    { label: 'WASM', value: 'wasm', disabled: false },
    { label: 'WebGPU', value: 'webgpu', disabled: !isWebGPUEnabled },
    { label: 'WebNN', value: 'webnn', disabled: !isWebNNEnabled },
  ];

  return (
    <div className="label flex flex-col items-start gap-2">
      <span> API:</span>
      <div className="join border border-gray-500 rounded-full">
        {options.map(({ label, value, disabled }) => (
          <button
            disabled={disabled}
            key={label}
            className={`btn join-item rounded-full px-6 ${
              selectedDevice === value ? 'bg-neutral text-white' : 'bg-base-100'
            }`}
            onClick={() => setSelectedDevice(value)}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
};

const WebNNDeviceSelector = ({
  selectedWebNNDevice,
  setSelectedWebNNDevice,
  isWebNNSelected,
  isNPUEnabled,
}) => {
  const options = [
    { label: 'Default', value: '', disabled: !isWebNNSelected },
    { label: 'CPU', value: 'cpu', disabled: !isWebNNSelected },
    { label: 'GPU', value: 'gpu', disabled: !isWebNNSelected },
    { label: 'NPU', value: 'npu', disabled: isWebNNSelected || !isNPUEnabled },
  ];

  return (
    <>
      <div className="label flex flex-col items-start gap-2">
        <span> WebNN Device:</span>
        <div className="join border border-gray-500 rounded-full">
          {options.map(({ label, value, disabled }) => (
            <button
              disabled={disabled}
              key={label}
              className={`btn join-item rounded-full px-6 ${
                selectedWebNNDevice === value
                  ? 'bg-neutral text-white'
                  : 'bg-base-100'
              }`}
              onClick={() => setSelectedWebNNDevice(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </>
  );
};

const QuantizationSelector = ({
  selectedQuantization,
  setSelectedQuantization,
  isFp16Enabled,
}) => {
  const options = [
    { label: 'q4', value: 'q4', disabled: false },
    { label: 'fp16', value: 'fp16', disabled: !isFp16Enabled },
    { label: 'fp32', value: 'fp32', disabled: false },
  ];

  return (
    <>
      <div className="label flex flex-col items-start gap-2">
        <span> Quantization:</span>
        <div className="join border border-gray-500 rounded-full">
          {options.map(({ label, value, disabled }) => (
            <button
              disabled={disabled}
              key={label}
              className={`btn join-item rounded-full px-6 ${
                selectedQuantization === value
                  ? 'bg-neutral text-white'
                  : 'bg-base-100'
              }`}
              onClick={() => setSelectedQuantization(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </>
  );
};

const SampleTabs = ({
  selectedTab,
  setSelectedTab,
  handleFileUpload,
  selectedImageUrl,
}) => {
  return (
    <>
      <div className="tabs tabs-bordered tabs-lg w-full ">
        {sampleImages.map((image, index) => (
          <button
            key={index}
            className={`tab tab-lg ${
              selectedTab === index ? 'tab-active' : ''
            }`}
            onClick={() => setSelectedTab(index)}
          >
            {image.name}
          </button>
        ))}
      </div>
      <div className="w-full flex flex-col items-center">
        {selectedTab !== 4 && (
          <img
            src={sampleImages[selectedTab].url}
            alt={sampleImages[selectedTab].name}
            className="w-full max-w-lg rounded-lg"
          />
        )}
        {selectedTab === 4 && (
          <div className="flex flex-col items-center gap-4">
            <input
              type="file"
              className="file-input w-full max-w-lg"
              onChange={handleFileUpload}
            />
            {selectedImageUrl && (
              <img
                src={selectedImageUrl}
                alt="Uploaded"
                className="w-full max-w-lg rounded-lg"
              />
            )}
          </div>
        )}
      </div>
    </>
  );
};

const ModelSelector = ({ selectedModel, setSelectedModel }) => {
  const options = [
    {
      label: 'Xenova/fastvit_t12.apple_in1k',
      value: 'Xenova/fastvit_t12.apple_in1k',
    },
    {
      label: 'Xenova/resnet-50',
      value: 'Xenova/resnet-50',
    },
  ];

  return (
    <label className=" label flex flex-col items-start gap-2 ">
      <span> Model:</span>
      <select
        className="select select-accent w-full max-w-xs  border border-gray-500 rounded-full"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        {options.map(({ label, value }) => (
          <option key={label} value={value}>
            {label}
          </option>
        ))}
      </select>
    </label>
  );
};
