import os
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import cv2


class YOLOv5Calibrator(trt.IInt8Calibrator):
    def __init__(self, calib_data_dir, input_shape, batch_size=8, cache_file="calibration.cache"):
        super().__init__()
        self.calib_data = []
        self.batch_size = batch_size
        self.input_shape = input_shape  # (C, H, W)
        self.cache_file = cache_file
        self.current_idx = 0
        self.device_input = None

        # Load calibration images
        self._load_calibration_images(calib_data_dir)

        # Allocate device memory
        if len(self.calib_data) > 0:
            self.device_input = cuda.mem_alloc(self.calib_data[0].nbytes * self.batch_size)

    def _load_calibration_images(self, calib_data_dir):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        calib_files = [os.path.join(calib_data_dir, f)
                       for f in os.listdir(calib_data_dir)
                       if any(f.lower().endswith(ext) for ext in img_extensions)]

        print(f"Found {len(calib_files)} calibration images")

        for img_path in calib_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load image: {img_path}")
                continue

            # Resize and preprocess
            img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = np.ascontiguousarray(img) / 255.0  # Normalize to [0, 1]
            img = img.astype(np.float32)

            self.calib_data.append(img)

            if len(self.calib_data) >= 100:  # Limit to 100 images for calibration
                break

        print(f"Loaded {len(self.calib_data)} images for calibration")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= len(self.calib_data):
            return None

        batch_end = min(self.current_idx + self.batch_size, len(self.calib_data))
        batch_size = batch_end - self.current_idx

        # Create a batch
        batch = np.zeros((batch_size, *self.input_shape), dtype=np.float32)
        for i in range(batch_size):
            batch[i] = self.calib_data[self.current_idx + i]

        # Copy to device
        cuda.memcpy_htod(self.device_input, batch)
        self.current_idx += batch_size

        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def get_algorithm(self):
        """Returns the calibration algorithm type."""
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2  # This is commonly used for INT8 calibration

    def __del__(self):
        if self.device_input:
            self.device_input.free()



def build_engine(onnx_path, engine_path, precision="fp32", workspace=4,
                 dynamic=True, batch_size=1, input_shape=(3, 640, 640),
                 calib_data_dir=None, verbose=False):
    """
    Build a TensorRT engine from an ONNX file

    Args:
        onnx_path (str): Path to the ONNX model
        engine_path (str): Path to save the TensorRT engine
        precision (str): Precision mode - 'fp32', 'fp16', or 'int8'
        workspace (int): Workspace size in GB
        dynamic (bool): Use dynamic input shapes
        batch_size (int): Max batch size for dynamic shapes
        input_shape (tuple): Input shape (C, H, W)
        calib_data_dir (str): Directory containing calibration images for INT8
        verbose (bool): Enable verbose logging

    Returns:
        str: Path to the generated engine file
    """
    print(f"Building TensorRT engine from {onnx_path}")
    print(f"Precision: {precision}, Workspace: {workspace}GB")

    # Create logger
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    # Create builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))

    # Create network
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parser error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX file")

    # Set input shape
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        print(f"Input '{inp.name}' with shape {inp.shape} and dtype {inp.dtype}")

    for out in outputs:
        print(f"Output '{out.name}' with shape {out.shape} and dtype {out.dtype}")

    # Configure dynamic shapes if needed
    if dynamic:
        profile = builder.create_optimization_profile()

        for inp in inputs:
            min_shape = (1, *input_shape)
            opt_shape = (batch_size // 2, *input_shape)
            max_shape = (batch_size, *input_shape)

            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
            print(f"Setting dynamic shape for {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

        config.add_optimization_profile(profile)

    # Set precision
    if precision == "fp16" and builder.platform_has_fast_fp16:
        print("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        print("Enabling INT8 precision")
        config.set_flag(trt.BuilderFlag.INT8)

        if calib_data_dir:
            print(f"Using calibration data from {calib_data_dir}")
            calibrator = YOLOv5Calibrator(
                calib_data_dir=calib_data_dir,
                input_shape=input_shape,
                batch_size=batch_size,
                cache_file=f"{Path(engine_path).stem}_calibration.cache"
            )
            config.int8_calibrator = calibrator
        else:
            raise ValueError("INT8 precision requires calibration data")

    # Build engine
    print("Building engine...")
    with builder.build_serialized_network(network, config) as engine:
        with open(engine_path, 'wb') as f:
            f.write(engine)

    print(f"Engine saved to {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv5 ONNX model to TensorRT engine")
    parser.add_argument("--onnx", required=True, help="Path to the ONNX model")
    parser.add_argument("--engine", required=True, help="Path to save the TensorRT engine")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16", help="Precision mode")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size in GB")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic input shapes")
    parser.add_argument("--batch_size", type=int, default=1, help="Max batch size for dynamic shapes")
    parser.add_argument("--height", type=int, default=640, help="Input height")
    parser.add_argument("--width", type=int, default=640, help="Input width")
    parser.add_argument("--calib_data", help="Directory with calibration images (required for INT8)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Check if calibration data is provided for INT8
    if args.precision == "int8" and not args.calib_data:
        raise ValueError("INT8 precision requires calibration data. Use --calib_data argument.")

    # Make sure the input ONNX file exists
    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.engine)), exist_ok=True)

    # Build the engine
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        workspace=args.workspace,
        dynamic=args.dynamic,
        batch_size=args.batch_size,
        input_shape=(3, args.height, args.width),
        calib_data_dir=args.calib_data,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()