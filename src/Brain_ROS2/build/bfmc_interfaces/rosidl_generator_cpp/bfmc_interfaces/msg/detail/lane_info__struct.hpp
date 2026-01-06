// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from bfmc_interfaces:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__STRUCT_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__bfmc_interfaces__msg__LaneInfo __attribute__((deprecated))
#else
# define DEPRECATED__bfmc_interfaces__msg__LaneInfo __declspec(deprecated)
#endif

namespace bfmc_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct LaneInfo_
{
  using Type = LaneInfo_<ContainerAllocator>;

  explicit LaneInfo_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->offset = 0l;
      this->detected = false;
      this->left_found = false;
      this->right_found = false;
    }
  }

  explicit LaneInfo_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->offset = 0l;
      this->detected = false;
      this->left_found = false;
      this->right_found = false;
    }
  }

  // field types and members
  using _offset_type =
    int32_t;
  _offset_type offset;
  using _detected_type =
    bool;
  _detected_type detected;
  using _left_found_type =
    bool;
  _left_found_type left_found;
  using _left_coeffs_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _left_coeffs_type left_coeffs;
  using _right_found_type =
    bool;
  _right_found_type right_found;
  using _right_coeffs_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _right_coeffs_type right_coeffs;

  // setters for named parameter idiom
  Type & set__offset(
    const int32_t & _arg)
  {
    this->offset = _arg;
    return *this;
  }
  Type & set__detected(
    const bool & _arg)
  {
    this->detected = _arg;
    return *this;
  }
  Type & set__left_found(
    const bool & _arg)
  {
    this->left_found = _arg;
    return *this;
  }
  Type & set__left_coeffs(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->left_coeffs = _arg;
    return *this;
  }
  Type & set__right_found(
    const bool & _arg)
  {
    this->right_found = _arg;
    return *this;
  }
  Type & set__right_coeffs(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->right_coeffs = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    bfmc_interfaces::msg::LaneInfo_<ContainerAllocator> *;
  using ConstRawPtr =
    const bfmc_interfaces::msg::LaneInfo_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<bfmc_interfaces::msg::LaneInfo_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<bfmc_interfaces::msg::LaneInfo_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      bfmc_interfaces::msg::LaneInfo_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<bfmc_interfaces::msg::LaneInfo_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      bfmc_interfaces::msg::LaneInfo_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<bfmc_interfaces::msg::LaneInfo_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<bfmc_interfaces::msg::LaneInfo_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<bfmc_interfaces::msg::LaneInfo_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__bfmc_interfaces__msg__LaneInfo
    std::shared_ptr<bfmc_interfaces::msg::LaneInfo_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__bfmc_interfaces__msg__LaneInfo
    std::shared_ptr<bfmc_interfaces::msg::LaneInfo_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const LaneInfo_ & other) const
  {
    if (this->offset != other.offset) {
      return false;
    }
    if (this->detected != other.detected) {
      return false;
    }
    if (this->left_found != other.left_found) {
      return false;
    }
    if (this->left_coeffs != other.left_coeffs) {
      return false;
    }
    if (this->right_found != other.right_found) {
      return false;
    }
    if (this->right_coeffs != other.right_coeffs) {
      return false;
    }
    return true;
  }
  bool operator!=(const LaneInfo_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct LaneInfo_

// alias to use template instance with default allocator
using LaneInfo =
  bfmc_interfaces::msg::LaneInfo_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace bfmc_interfaces

#endif  // BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__STRUCT_HPP_
