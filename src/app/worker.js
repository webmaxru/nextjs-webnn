import { pipeline, env } from '@huggingface/transformers';

// Skip local model check
env.allowLocalModels = false;

env.backends.onnx.debug = true;
// env.backends.onnx.logLevel = 'verbose';

env.backends.onnx.wasm.simd = false;

// Use the Singleton pattern to enable lazy construction of the pipeline.
class PipelineSingleton {
  static instance = null;
  static task = null;
  static model = null;
  static device = null;
  static dtype = null;
  static session_options = null;

  static async getInstance(
    task,
    model,
    device,
    dtype,
    session_options,
    progress_callback = null
  ) {
    if (
      this.instance === null ||
      this.model !== model ||
      this.device !== device ||
      this.dtype !== dtype ||
      this.session_options !== session_options ||
      this.task !== task
    ) {
      console.log(
        `[Worker] Creating pipeline with task ${task}, model ${model}, device ${device}, dtype ${dtype}, session options ${JSON.stringify(session_options)}`
      );
      this.instance = await pipeline(task, model, {
        progress_callback,
        device: device,
        dtype: dtype,
        session_options: session_options,
      });
      this.model = model;
      this.device = device;
      this.dtype = dtype;
      this.session_options = session_options;
      this.task = task;
    }
    return this.instance;
  }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
  const { task, input, model, device, dtype, session_options } = event.data;

  // Retrieve the classification pipeline. When called for the first time,
  // this will load the pipeline and save it for future use.
  let classifier = await PipelineSingleton.getInstance(
    task,
    model,
    device,
    dtype,
    session_options,
    (x) => {
      // We also add a progress callback to the pipeline so that we can
      // track model loading.
      self.postMessage(x);
    }
  );

  // Actually perform the classification
  let output = await classifier(input);

  console.log(`[Worker] Completed`);
  // Send the output back to the main thread
  self.postMessage({
    status: 'complete',
    output: output,
  });
});
