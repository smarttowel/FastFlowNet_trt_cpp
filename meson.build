project('flownet', 'cpp', 'cuda',
  version : '0.1',
  default_options : ['warning_level=2',
                     'cpp_std=c++17'])

cpp = meson.get_compiler('cpp')

cuda = import('unstable-cuda')
arch_flags = cuda.nvcc_arch_flags(meson.get_compiler('cuda'), '7.5')
add_project_arguments('--use_fast_math', 
                      '--expt-relaxed-constexpr', 
                      '--default-stream', 'per-thread',
                       arch_flags,
                       language: 'cuda')

nvinfer_dep = [cpp.find_library('nvinfer', dirs: ['/home/pavel/.local/lib/python3.10/site-packages/tensorrt/']),
               cpp.find_library('nvonnxparser', dirs: ['/home/pavel/.local/lib/python3.10/site-packages/tensorrt/'])]
cuda_dep = [dependency('cuda-10.2'),
            dependency('cudart-10.2')]
opencv_dep = dependency('opencv')

exe = executable('flownet', 
                     'main.cpp',
                     'correlationPlugin.cpp',
                     'correlation.cu',
                     'gridSamplerPlugin.cpp',
                     'gridSampler.cu',
                      cuda_args: arch_flags,
                      dependencies: [cuda_dep,
                                     nvinfer_dep,
                                     opencv_dep])
