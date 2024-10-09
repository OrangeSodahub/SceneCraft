// Copyright (c) OpenMMLab. All rights reserved.
#include <torch/extension.h>
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"


// cudabind

int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);

int NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);

void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3);

int hard_voxelize_forward_cuda(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim) {
  return HardVoxelizeForwardCUDAKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
};

int nondeterministic_hard_voxelize_forward_cuda(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim) {
  return NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
};

void dynamic_voxelize_forward_cuda(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim) {
  DynamicVoxelizeForwardCUDAKernelLauncher(points, coors, voxel_size,
                                           coors_range, NDim);
};

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim = 3) {
  return DISPATCH_DEVICE_IMPL(hard_voxelize_forward_impl, points, voxels, coors,
                              num_points_per_voxel, voxel_size, coors_range,
                              max_points, max_voxels, NDim);
}

int nondeterministic_hard_voxelize_forward_impl(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  return DISPATCH_DEVICE_IMPL(nondeterministic_hard_voxelize_forward_impl,
                              points, voxels, coors, num_points_per_voxel,
                              voxel_size, coors_range, max_points, max_voxels,
                              NDim);
}

void dynamic_voxelize_forward_impl(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim = 3) {
  DISPATCH_DEVICE_IMPL(dynamic_voxelize_forward_impl, points, coors, voxel_size,
                       coors_range, NDim);
}


REGISTER_DEVICE_IMPL(hard_voxelize_forward_impl, CUDA,
                     hard_voxelize_forward_cuda);
REGISTER_DEVICE_IMPL(nondeterministic_hard_voxelize_forward_impl, CUDA,
                     nondeterministic_hard_voxelize_forward_cuda);
REGISTER_DEVICE_IMPL(dynamic_voxelize_forward_impl, CUDA,
                     dynamic_voxelize_forward_cuda);


// module api

void hard_voxelize_forward(const at::Tensor &points,
                           const at::Tensor &voxel_size,
                           const at::Tensor &coors_range, at::Tensor &voxels,
                           at::Tensor &coors, at::Tensor &num_points_per_voxel,
                           at::Tensor &voxel_num, const int max_points,
                           const int max_voxels, const int NDim = 3,
                           const bool deterministic = true) {
  int64_t *voxel_num_data = voxel_num.data_ptr<int64_t>();
  std::vector<float> voxel_size_v(
      voxel_size.data_ptr<float>(),
      voxel_size.data_ptr<float>() + voxel_size.numel());
  std::vector<float> coors_range_v(
      coors_range.data_ptr<float>(),
      coors_range.data_ptr<float>() + coors_range.numel());

  if (deterministic) {
    *voxel_num_data = hard_voxelize_forward_impl(
        points, voxels, coors, num_points_per_voxel, voxel_size_v,
        coors_range_v, max_points, max_voxels, NDim);
  } else {
    *voxel_num_data = nondeterministic_hard_voxelize_forward_impl(
        points, voxels, coors, num_points_per_voxel, voxel_size_v,
        coors_range_v, max_points, max_voxels, NDim);
  }
}

void dynamic_voxelize_forward(const at::Tensor &points,
                              const at::Tensor &voxel_size,
                              const at::Tensor &coors_range, at::Tensor &coors,
                              const int NDim = 3) {
  std::vector<float> voxel_size_v(
      voxel_size.data_ptr<float>(),
      voxel_size.data_ptr<float>() + voxel_size.numel());
  std::vector<float> coors_range_v(
      coors_range.data_ptr<float>(),
      coors_range.data_ptr<float>() + coors_range.numel());
  dynamic_voxelize_forward_impl(points, coors, voxel_size_v, coors_range_v,
                                NDim);
}


// pybind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hard_voxelize_forward", &hard_voxelize_forward,
        "hard_voxelize_forward", py::arg("points"), py::arg("voxel_size"),
        py::arg("coors_range"), py::arg("voxels"), py::arg("coors"),
        py::arg("num_points_per_voxel"), py::arg("voxel_num"),
        py::arg("max_points"), py::arg("max_voxels"), py::arg("NDim"),
        py::arg("deterministic"));
  m.def("dynamic_voxelize_forward", &dynamic_voxelize_forward,
        "dynamic_voxelize_forward", py::arg("points"), py::arg("voxel_size"),
        py::arg("coors_range"), py::arg("coors"), py::arg("NDim"));
}