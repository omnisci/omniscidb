/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef HAVE_TBB
#include "tbb/concurrent_vector.h"
#include "tbb/task_group.h"
#endif

#include <future>
#include <iostream>
#include <type_traits>

namespace threadpool {

template <typename T>
class FuturesThreadPoolBase {
 public:
  template <class Function, class... Args>
  void spawn(Function&& f, Args&&... args) {
    threads_.push_back(std::async(std::launch::async, f, args...));
  }

 protected:
  std::vector<std::future<T>> threads_;
};

template <typename T, typename ENABLE = void>
class FuturesThreadPool : public FuturesThreadPoolBase<T> {
 public:
  FuturesThreadPool() {}

  void join() {
    for (auto& child : this->threads_) {
      child.wait();
    }
    for (auto& child : this->threads_) {
      child.get();
    }
  }
};

template <typename T>
class FuturesThreadPool<T, std::enable_if_t<std::is_object<T>::value>>
    : public FuturesThreadPoolBase<T> {
 public:
  FuturesThreadPool() {}

  auto join() {
    std::vector<T> results;
    results.reserve(this->threads_.size());
    for (auto& child : this->threads_) {
      child.wait();
    }
    for (auto& child : this->threads_) {
      results.push_back(child.get());
    }
    return results;
  }
};

#ifdef HAVE_TBB

class TbbThreadPoolBase {
 protected:
  tbb::task_group tasks_;
};

template <typename T, typename ENABLE = void>
class TbbThreadPool : public TbbThreadPoolBase {
 public:
  TbbThreadPool() {}

  template <class Function, class... Args>
  void spawn(Function&& f, Args&&... args) {
    tasks_.run([f, args...] { f(args...); });
  }

  void join() { tasks_.wait(); }
};

template <typename T>
class TbbThreadPool<T, std::enable_if_t<std::is_object<T>::value>>
    : public TbbThreadPoolBase {
 public:
  TbbThreadPool() {}

  template <class Function, class... Args>
  void spawn(Function&& f, Args&&... args) {
    const size_t result_idx = results_.size();
    results_.emplace_back(T{});
    tasks_.run([this, result_idx, f, args...] { results_[result_idx] = f(args...); });
  }

  auto join() {
    tasks_.wait();
    return results_;
  }

 private:
  tbb::concurrent_vector<T> results_;
};

#endif

#ifdef HAVE_TBB
template <typename T>
using ThreadPool = TbbThreadPool<T>;
#else
template <typename T>
using ThreadPool = FuturesThreadPool<T>;
#endif
};  // namespace threadpool
