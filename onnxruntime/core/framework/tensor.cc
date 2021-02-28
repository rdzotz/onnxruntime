// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"

#include <utility>
#include "core/common/safeint.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/data_types.h"

namespace onnxruntime {

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc,
               ptrdiff_t offset)
    : alloc_info_(alloc) {
  ORT_ENFORCE(p_type != nullptr);
  Init(p_type, shape, p_data, offset);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, AllocatorPtr&& deleter)
    : alloc_info_(deleter->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  ORT_ENFORCE(deleter);
  deleters_.push_back(CreateBufDelClr(std::move(deleter), p_data));
  Init(p_type, shape, p_data, 0);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, ptrdiff_t offset,
               const OrtMemoryInfo& alloc, std::vector<std::function<void(void)>>&& deleters)
    : alloc_info_(alloc), deleters_(std::move(deleters)) {
  ORT_ENFORCE(p_type != nullptr);
  Init(p_type, shape, p_data, offset);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator)
    : alloc_info_(allocator->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  int64_t shape_size = shape.Size();  // value returned is checked for overflow by TensorShape::Size()
  if (shape_size < 0)
    ORT_THROW("shape.Size() must >=0");

  void* p_data = nullptr;
  if (shape_size > 0) {
    SafeInt<size_t> len = 0;
    if (!allocator->CalcMemSizeForArray(SafeInt<size_t>(shape_size), p_type->Size(), &len))
      ORT_THROW("tensor failed memory size calculation");

    p_data = allocator->Alloc(len);
  }

  // for string tensors, do the placement new for strings on pre-allocated buffer.
  if (utils::IsPrimitiveDataType<std::string>(p_type->AsPrimitiveDataType())) {
    auto* ptr = static_cast<std::string*>(p_data);
    for (int64_t i = 0, n = shape_size; i < n; ++i) {
      new (ptr + i) std::string();
    }
  }

  // This tensor own the buffer, setup release the buffer when this tensor is deconstructed.
  if (utils::IsPrimitiveDataType<std::string>(p_type->AsPrimitiveDataType())) {
     deleters_.push_back([p_data, shape_size]() {
        using string = std::string;
        auto* ptr = static_cast<std::string*>(p_data);
        int64_t len = shape_size;
        for (int64_t i = 0; i < len; i++)
          ptr[i].~string();
     });
  }
  deleters_.push_back(CreateBufDelClr(std::move(allocator), p_data));

  Init(p_type, shape, p_data, 0);
}

size_t Tensor::SizeInBytes() const {
  size_t ret;
  if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(shape_.Size()), dtype_->Size(), &ret)) {
    ORT_THROW("tensor size overflow");
  }
  return ret;
}

void Tensor::Init(MLDataType p_type, const TensorShape& shape, void* p_raw_data, ptrdiff_t offset) {
  int64_t shape_size = shape.Size();
  if (shape_size < 0) ORT_THROW("shape.Size() must >=0");
  dtype_ = p_type->AsPrimitiveDataType();
  ORT_ENFORCE(dtype_ != nullptr, "Tensor is expected to contain one of the primitive data types. Got: ",
              DataTypeImpl::ToString(p_type));
  shape_ = shape;
  p_data_ = p_raw_data;
  byte_offset_ = offset;
}

Tensor::Tensor(Tensor&& other) noexcept
    : p_data_(other.p_data_),
      shape_(other.shape_),
      dtype_(other.dtype_),
      alloc_info_(other.alloc_info_),
      byte_offset_(other.byte_offset_),
      deleters_(std::move(other.deleters_)) {
  other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
  other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
  other.p_data_ = nullptr;
  other.deleters_.clear();
  other.byte_offset_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    ReleaseBuffer();

    dtype_ = other.dtype_;
    shape_ = other.shape_;
    alloc_info_ = other.alloc_info_;
    byte_offset_ = other.byte_offset_;
    p_data_ = other.p_data_;
    deleters_ = std::move(other.deleters_);

    other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
    other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
    other.p_data_ = nullptr;
    other.byte_offset_ = 0;
    other.deleters_.clear();
  }
  return *this;
}

Tensor::~Tensor() {
  ReleaseBuffer();
}

void Tensor::ReleaseBuffer() {
  auto dit = deleters_.begin();
  while (dit != deleters_.end()) {
    (*dit)();
    deleters_.erase(dit);
    dit = deleters_.begin();
  }
}

}  // namespace onnxruntime
