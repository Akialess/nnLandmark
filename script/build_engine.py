import os, sys, time
import argparse
import tensorrt as trt

# https://github.com/NVIDIA/TensorRT/blob/main/samples/python/refactored/1_run_onnx_with_tensorrt/main.ipynb
def build_engine(onnx_path, engine_path, input_shape):

    input_shape = tuple(int(x) for x in input_shape.split(","))

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Bind the TensorRT network to the parser so that the parser can update the network later accordingly
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX model at {onnx_path}")
    with open(onnx_path, "rb") as model:
        parser.parse(model.read())
    
    print('Parsing ONNX model... done')

    config = builder.create_builder_config()

    # TensorRT needs memory for layer operations and intermediate activations during inference
    # Setting a memory limit helps control resource usage and prevents out-of-memory errors
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30) 

    # Optimization
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    print('Starting to build engine. This might take several minutes depending on the hardware...')
    engine = builder.build_serialized_network(network, config)
    assert engine is not None, 'Engine build failed'

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)

    with open(engine_path, 'wb') as f:
        f.write(engine)

    print("TensorRT engine created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT Engine")
    parser.add_argument("--onnx", required=True, help="Path to input ONNX model")
    parser.add_argument("--engine", required=True, help="Path to output TRT engine")
    parser.add_argument("--input_shape", required=True, help="Input shape, e.g., 1,1,96,64,48")
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, args.input_shape)
